import cv2.cv2 as cv2
import numpy as np
import mediapipe as mp
import pandas as pd

import os


def find_bounding_box(coords):
    x_coords, y_coords = zip(*coords)
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def create_skeleton(image_path, _out_dir, _out_path, _gestures, _gesture_name, _mp_hands, _hands, _mp_draw):
    upscale_factor = 4

    img = cv2.flip(cv2.imread(image_path), 1)
    img = cv2.resize(img, (int(img.shape[1] * upscale_factor), int(img.shape[0] * upscale_factor)), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    results = _hands.process(img_rgb)
    out_path_csv = os.path.join(_out_dir, _out_path)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            is_right_handed = True if results.multi_handedness[0].classification[0].label == 'Right' else False

            lm_points = []
            for landmark in hand.landmark:
                lm_points.append(landmark.x)
                lm_points.append(landmark.y)

            lm_points_2d = np.reshape(lm_points, (len(hand.landmark), 2))
            bounding_box = find_bounding_box(lm_points_2d)

            lm_points_normalized = []
            for x, y in lm_points_2d:
                norm_x = (x - bounding_box[0]) / (bounding_box[2] - bounding_box[0])
                norm_y = (y - bounding_box[1]) / (bounding_box[3] - bounding_box[1])
                lm_points_normalized.append(norm_x if is_right_handed else 1 - norm_x)
                lm_points_normalized.append(norm_y)
            for i in range(len(_gestures)):
                lm_points_normalized.append(1 if i == _gestures[_gesture_name] else 0)

            df = pd.DataFrame([lm_points_normalized])
            df.to_csv(out_path_csv, mode='a', index=False, header=not os.path.exists(out_path_csv))

            img_skeleton = np.zeros(img.shape)
            _mp_draw.draw_landmarks(img_skeleton, hand, _mp_hands.HAND_CONNECTIONS)
            bounding_box = [int(bounding_box[0] * w), int(bounding_box[1] * h), int(bounding_box[2] * w), int(bounding_box[3] * h)]
            bounding_box = np.clip(bounding_box, 0, max(h, w) * upscale_factor)
            bounding_box_x_len, bounding_box_y_len = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
            return img_skeleton[bounding_box[1]:bounding_box[3] + 1, bounding_box[0]:bounding_box[2] + 1], bounding_box_x_len + 1, bounding_box_y_len + 1
    else:
        return None, None, None


def dfs_padding(cur_path, _x_len, _y_len):
    print(f'Currently padding: {cur_path}')
    try:
        sub_dirs = os.listdir(cur_path)
        for sub_dir in sub_dirs:
            dfs_padding(os.path.join(cur_path, sub_dir), _x_len, _y_len)
    except NotADirectoryError:
        cur_dirs = cur_path.split(os.path.sep)
        if cur_dirs[-1].split('.')[-1] in ['png', 'jpg', 'jpeg']:
            img = cv2.imread(cur_path)
            h, w = img.shape[:2]
            top, left = (_y_len - h) // 2, (_x_len - w) // 2
            bot, right = _y_len - h - top, _x_len - w - left
            img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))
            cv2.imwrite(cur_path, img)


def dfs_output(cur_path, prev_path, _out_dir, _out_path, _gestures, _mp_hands, _hands, _mp_draw):
    print(f'Currently processing: {cur_path}')
    _x_len, _y_len = 0, 0
    try:
        sub_dirs = os.listdir(cur_path)
        for sub_dir in sub_dirs:
            x, y = dfs_output(os.path.join(cur_path, sub_dir), cur_path, _out_dir, _out_path, _gestures, _mp_hands, _hands, _mp_draw)
            if _x_len < x:
                _x_len = x
            if _y_len < y:
                _y_len = y
    except NotADirectoryError:
        cur_dirs, prev_dirs = cur_path.split(os.path.sep), prev_path.split(os.path.sep)
        if cur_dirs[-1].split('.')[-1] in ['png', 'jpg', 'jpeg']:
            cur_dirs[0] = prev_dirs[0] = out_dir
            os.makedirs(os.path.join(*prev_dirs), exist_ok=True)
            img_skeleton, bounding_box_x_len, bounding_box_y_len = create_skeleton(cur_path, _out_dir, _out_path, _gestures, prev_dirs[-1], _mp_hands, _hands, _mp_draw)
            if img_skeleton is not None:
                cv2.imwrite(os.path.join(*cur_dirs), img_skeleton)
                return bounding_box_x_len, bounding_box_y_len
        else:
            return -1, -1
    return _x_len, _y_len


if __name__ == '__main__':
    in_dir = 'input'
    out_dir = 'output'
    out_path = 'data.csv'

    gestures = {'A': 0, 'B': 1, 'C': 2}  # 0 = Left Click, 1 = Right Click, 2 = Middle Click

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    x_len, y_len = dfs_output(in_dir, str(), out_dir, out_path, gestures, mp_hands, hands, mp_draw)
    dfs_padding(out_dir, x_len, y_len)
