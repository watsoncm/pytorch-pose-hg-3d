import cv2
import numpy as np
import torch
import time
import _init_paths
from opts import opts
from model import create_model
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d
from utils.debugger import Debugger


mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

opt = opts().parse()
opt.heads['depth'] = opt.num_output
cap = cv2.VideoCapture(0)
model, _, _ = create_model(opt)
model = model.to('cpu')
model.eval()


def get_angle(first, vertex, second):
  first_diff = first - vertex
  second_diff = second - vertex
  first_norm = np.linalg.norm(first_diff)
  second_norm = np.linalg.norm(second_diff)
  cos_theta = first_diff.dot(second_diff) / (first_norm * second_norm)
  return np.arccos(cos_theta)


def get_score_right_hand_to_mouth(pred_3d):
    score = 0
    right_hand_pos = pred_3d[10, :]
    right_shoulder_pos = pred_3d[12, :]
    left_shoulder_pos = pred_3d[13, :]
    neck_base_pos = pred_3d[8, :]
    face_center_pos = pred_3d[9, :]

    average_shoulder_pos = (right_shoulder_pos[0] + left_shoulder_pos[0]) * 0.5
    dist_between_shoulders = abs(right_shoulder_pos[0] - left_shoulder_pos[0])
    distance_from_midline = abs(abs(right_hand_pos[0] - left_shoulder_pos[0]) - abs(right_hand_pos[0] - average_shoulder_pos))

    dist_above_neck = right_hand_pos[1] - neck_base_pos[1]
    dist_below_face = face_center_pos[1] - right_hand_pos[1]

    if right_hand_pos[0] > right_shoulder_pos[0] or right_shoulder_pos[1] < right_hand_pos[1]:
        score = 0.5
    elif right_hand_pos[0] > right_shoulder_pos[0] and right_shoulder_pos[1] < right_hand_pos[1]:
        score = 1
    elif distance_from_midline < 0.15 * dist_between_shoulders and dist_below_face > dist_above_neck:
        score = 2

    return score



while True:
    _, image = cap.read()

    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
      c, s, 0, [opt.input_w, opt.input_h])
    inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to('cpu')
    out = model(inp)[-1]
    pred = get_preds(out['hm'].detach().cpu().numpy())[0]
    pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
                           out['depth'].detach().cpu().numpy())[0]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
