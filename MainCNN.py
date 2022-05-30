import Preprocessing
import Util

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import mouse
import ctypes
import win32api
import win32con
import time

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

prev_time = time.time()

gestures = ['Left Click', 'Right Click', 'Middle Click', 'Neutral']
num_gestures = len(gestures)
saved_gesture_index = 0

model = tf.keras.models.load_model('model/model_cnn_seg.h5')

time_passed_since_last_prediction = 1
prediction_interval = 1
is_scrolling = False


while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            h, w, c = img.shape
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            lm_points = []
            for landmark in hand.landmark:
                lm_points.append(landmark.x)
                lm_points.append(landmark.y)

            saved_frame_2d = np.reshape(lm_points, (len(hand.landmark), 2))
            bb = Util.find_bounding_box(saved_frame_2d)

            bb = [(len(img_rgb[0]) * bb[0]), (len(img_rgb) * bb[1]), (len(img_rgb[0]) * bb[2]), (len(img_rgb) * bb[3])]
            bias_x, bias_y = (bb[2] - bb[0]) // 5, (bb[3] - bb[1]) // 5
            bb = [Util.clamp(bb[0] - bias_x, 0, w), Util.clamp(bb[1] - bias_y, 0, h), Util.clamp(bb[2] + bias_x, 0, w), Util.clamp(bb[3] + bias_y, 0, h)]

            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            img_bb = Preprocessing.segmentate_image_kmeans(img_gray)
            img_bb = img_bb[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

            img_colors = sorted(list(set(img_bb.flatten())))
            for i in range(len(img_bb)):
                img_bb[i] = np.where(img_bb[i] == img_colors[0], 0, img_bb[i])
                img_bb[i] = np.where(img_bb[i] == img_colors[1], 255, img_bb[i])

            img_bb = cv2.resize(img_bb, (128, 128))
            img_bb = cv2.GaussianBlur(img_bb, (3, 3), 0)
            img_bb = cv2.bitwise_not(img_bb)
            #cv2.imshow("Test", img_bb)
            #cv2.waitKey(1)

            predictions = model.predict([img_bb.tolist()])
            prediction, prediction_index = np.amax(predictions), np.argmax(predictions)
            cv2.putText(img, gestures[prediction_index] if prediction > 0.9 and prediction_index != 3 else 'Neutral', (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

            x, y = hand.landmark[0].x * 1.4 - 0.2, hand.landmark[1].y * 1.4 - 0.2
            mouse.move(x * ctypes.windll.user32.GetSystemMetrics(0), y * ctypes.windll.user32.GetSystemMetrics(1))
            if prediction > 0.9:
                if prediction_index == 0 and time_passed_since_last_prediction > prediction_interval:
                    if is_scrolling:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0)
                        is_scrolling = False
                    mouse.click('left')
                    time_passed_since_last_prediction = 0
                elif prediction_index == 1 and time_passed_since_last_prediction > prediction_interval:
                    if is_scrolling:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0)
                        is_scrolling = False
                    mouse.click('right')
                    time_passed_since_last_prediction = 0
                elif prediction_index == 2:
                    if not is_scrolling:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0)
                        is_scrolling = True
                    time_passed_since_last_prediction = 0
                else:
                    if is_scrolling:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0)
                        is_scrolling = False
                    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0)
            else:
                if is_scrolling:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0)
                    is_scrolling = False
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0)


    current_time = time.time()
    delta_time = current_time - prev_time
    fps = 1 / delta_time
    time_passed_since_last_prediction += delta_time
    cv2.putText(img, str(int(fps)), (580, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    prev_time = current_time
    cv2.imshow('Image', img)
    cv2.waitKey(1)
