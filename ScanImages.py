import cv2.cv2 as cv2
import numpy as np
import mediapipe as mp
import pandas as pd

import os

gestures = {'9': 0, '8': 1, '7': 2}  # 0 = Left Click, 1 = Right Click, 2 = Mouse Wheel

in_dir = 'input'
out_dir = 'output'
out_name = 'data.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.88)
mp_draw = mp.solutions.drawing_utils


def find_bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


in_names = os.listdir(in_dir)
for in_name in in_names:
    in_path = os.path.join(in_dir, in_name)
    out_path_csv = os.path.join(out_dir, out_name)
    out_path_png = os.path.join(out_dir, in_name)
    gesture_name = in_path.split('_')[1]

    if gesture_name in gestures.keys():
        img = cv2.flip(cv2.imread(in_path), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        results = hands.process(img_rgb)

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
                for i in range(len(gestures.keys())):
                    lm_points_normalized.append(1 if i == gestures[gesture_name] else 0)

                df = pd.DataFrame([lm_points_normalized])
                df.to_csv(out_path_csv, mode='a', index=False, header=not os.path.exists(out_path_csv))

                img_skeleton = np.zeros(img.shape)
                mp_draw.draw_landmarks(img_skeleton, hand, mp_hands.HAND_CONNECTIONS)
                bounding_box = [int(bounding_box[0] * w), int(bounding_box[1] * h), int(bounding_box[2] * w), int(bounding_box[3] * h)]
                cv2.imwrite(out_path_png, img_skeleton[bounding_box[1]:bounding_box[3] + 1, bounding_box[0]:bounding_box[2] + 1])
