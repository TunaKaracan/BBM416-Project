import Preprocessing
import Util

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
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
            #img_bb = cv2.bitwise_not(img_bb)
            cv2.imshow("Test", img_bb)
            cv2.waitKey(1)

            predictions = model.predict([img_bb.tolist()])
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
