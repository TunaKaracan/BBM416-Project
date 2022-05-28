import Preprocessing, Util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import os


lr = 1e-4
n_epochs = 50
n_batches = 128


def create_dataset(_hands):
    dataset = []

    # Preprocessing
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if dirpath is not input_dir:  # Ignoring the first path, as it's the folder containing the class sub-folders
            for i in filenames:
                path = dirpath + '\\' + i
                curr_loc = dirpath.split("\\")
                gesture = gestures[dirpath.split("\\")[-1]]

                img_orig = cv2.flip(cv2.imread(path), 1)
                results = hands.process(img_orig)

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        h, w, c = img_orig.shape
                        lm_points = []
                        for landmark in hand.landmark:
                            lm_points.append(landmark.x)
                            lm_points.append(landmark.y)

                        saved_frame_2d = np.reshape(lm_points, (len(hand.landmark), 2))
                        bb = Util.find_bounding_box(saved_frame_2d)
                        bb = [(len(img_orig[0]) * bb[0]), (len(img_orig) * bb[1]), (len(img_orig[0]) * bb[2]),
                              (len(img_orig) * bb[3])]
                        bias_x, bias_y = (bb[2] - bb[0]) // 5, (bb[3] - bb[1]) // 5
                        bb = [Util.clamp(bb[0] - bias_x, 0, w), Util.clamp(bb[1] - bias_y, 0, h),
                              Util.clamp(bb[2] + bias_x, 0, w), Util.clamp(bb[3] + bias_y, 0, h)]

                        img_seg = Preprocessing.segmentate_image_kmeans(img_orig)
                        img_seg = img_seg[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                        img_seg = cv2.resize(img_seg, (32, 32))
                        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(curr_loc[0] + "_seg/" + curr_loc[1] + "/" + i, img_seg)
                        dataset.append({"Name": i, "Class": gesture, "Image Data": img_seg.tolist()})

            print("Class {0} completed".format(gesture))

    return pd.DataFrame(dataset)


if __name__ == '__main__':
    input_dir = "input_new_2"
    gestures = {"A": 0, "B": 1, "C": 2}  # 0 = Left Click, 1 = Right Click, 2 = Middle Click

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.9)
    mp_draw = mp.solutions.drawing_utils

    gestures = {"A": 0, "B": 1, "C": 2}  # 0 = Left Click, 1 = Right Click, 2 = Middle Click

    df_img = create_dataset(hands)

    from sklearn.model_selection import train_test_split

    X = df_img["Image Data"].values.tolist()
    y = df_img["Class"].values.tolist()

    # Split the dataset into 60% train, %20 validation, 20% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    # X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
    # X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Dropout

    # Build the CNN architecture
    model = Sequential([
        # Convolutional layer
        Conv2D(32, (3, 3), input_shape=(len(X[0]), len(X[0][0]), 1), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.03375),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.0675),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding="same"),
        Dropout(0.125),
        Conv2D(192, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 5), strides=(3, 5), padding="same"),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),
        Dropout(0.5),

        # Fully connected layer
        Flatten(),
        Dense(64, activation="relu"),

        # Output layer
        Dense(len(gestures), activation="softmax")
    ])


    from tensorflow.keras.optimizers import Adam, RMSprop

    opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=n_epochs,
                        batch_size=n_batches,
                        validation_data=(X_test, y_test)
                        )

    from sklearn.metrics import classification_report, confusion_matrix

    print(model.evaluate(X_test, y_test))

    predictions = np.argmax(model.predict(X_test), axis=-1)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    model.summary()

    model.save("model/model_cnn_seg.h5")

    finishData = pd.DataFrame(history.history)
    fig, axs = plt.subplots(2)
    axs[0].plot(finishData["accuracy"], label="Train Accuracy")
    axs[0].plot(finishData["val_accuracy"], label="Validation Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_yticks(np.arange(0, 1, 0.1))
    axs[1].plot(finishData["loss"], label="Train Error")
    axs[1].plot(finishData["val_loss"], label="Validation Error")
    axs[1].legend(loc="upper right")
    axs[1].set_yticks(np.arange(2, 0, -0.2))
    plt.show()
