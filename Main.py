import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

import time

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.95, min_tracking_confidence=0.95)
mp_draw = mp.solutions.drawing_utils

prev_time = time.time()

gestures = ['Left Click', 'Right Click', 'Middle Click']
num_gestures = len(gestures)
saved_gesture_index = 0

model = tf.keras.models.load_model('model/model.h5')


def find_bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def rotate_point(point, angle, pivot=None):
    angle *= np.pi / 180
    if pivot is None:
        pivot = [0, 0]
    sin = np.sin(angle)
    cos = np.cos(angle)

    point = np.subtract(point, pivot)
    point = np.matmul([[cos, -sin], [sin, cos]], point)
    point = np.add(point, pivot)

    return list(point)


while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            is_right_handed = True if results.multi_handedness[0].classification[0].label == 'Right' else False
            h, w, c = img.shape
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            start_pos_1, end_pos_1 = (hand.landmark[5].x, hand.landmark[5].y), (hand.landmark[17].x, hand.landmark[17].y)
            start_pos_2, end_pos_2 = (hand.landmark[9].x, hand.landmark[9].y), (hand.landmark[0].x, hand.landmark[0].y)

            vector_1 = (end_pos_1[0] - start_pos_1[0], end_pos_1[1] - start_pos_1[1])
            vector_2 = (end_pos_2[0] - start_pos_2[0], end_pos_2[1] - start_pos_2[1])
            vector_3 = (vector_1[0] + vector_2[0], vector_1[1] + vector_2[1])

            lm_points = []
            for landmark in hand.landmark:
                lm_point = [landmark.x, 1 - landmark.y]
                lm_pivot = [hand.landmark[0].x, 1 - hand.landmark[0].y]
                lm_angle = -57 - np.arctan2(-vector_3[1], vector_3[0]) * 180 / np.pi
                res = rotate_point(lm_point, lm_angle, lm_pivot)
                lm_points.append(res[0])
                lm_points.append(1 - res[1])

            saved_frame_2d = np.reshape(lm_points, (len(hand.landmark), 2))
            bounding_box = find_bounding_box(saved_frame_2d)
            
            lm_points_normalized = []
            for x, y in saved_frame_2d:
                norm_x = (x - bounding_box[0]) / (bounding_box[2] - bounding_box[0])
                norm_y = (y - bounding_box[1]) / (bounding_box[3] - bounding_box[1])
                lm_points_normalized.append(norm_x if is_right_handed else 1 - norm_x)
                lm_points_normalized.append(norm_y)

            predictions = model.predict([lm_points_normalized])
            prediction, prediction_index = np.amax(predictions), np.argmax(predictions)
            if prediction > 0.9:
                cv2.putText(img, gestures[prediction_index], (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                print("Prediction Confidence:", prediction)
                print("Predicted Class:", prediction_index)
            else:
                cv2.putText(img, 'Neutral', (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    current_time = time.time()
    delta_time = 1 / (current_time - prev_time)
    cv2.putText(img, str(int(delta_time)), (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    prev_time = current_time
    cv2.imshow('Image', img)
    cv2.waitKey(1)
