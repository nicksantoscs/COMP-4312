# importing the libraries and frameworks into this project

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# this block of codes below here detects the hands
mp_hands = mp.solutions.holistic  # Hands model
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing Styles
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):

    # this block of codes below here converts the color to RGB and then back to BGR
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):

    # The RGB colors are opposite (BGR)

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(222, 31, 193), thickness=8, circle_radius=10), # circle
                              mp_drawing.DrawingSpec(color=(255, 128, 48), thickness=6, circle_radius=5) # line
                             )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(48, 128, 255), thickness=8, circle_radius=10), # circle
                              mp_drawing.DrawingSpec(color=(31, 50, 222), thickness=6, circle_radius=5) # line
                             )

def extract_keypoints(results):

    # it is extracting the keypoint values for both hands
    left = np.array([[res.x, res.y, res.z] for res in
                     results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    right = np.array([[res.x, res.y, res.z] for res in
                      results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return  np.concatenate([left,right])


# it is going to export the data and numpy arrays
DATA_PATH = os.path.join('ASL_Fingerspelling_Data')

# the program will identify which letter did the user signed
actions = np.array(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
     'X', 'Y', 'Z', 'ZZ'])

# 30 videos worth of data
no_sequences = 30

# the video is going to be 30 frames in length
sequence_length = 30

# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass


label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
model.summary()

# This block of codes below here perform a loop of capturing the frames and then extract them to the mediapipe
cap = cv2.VideoCapture(0)  # accessing the webcam which located at 0

# with mp_hands.Holistic(
#         model_complexity=0,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as holistic:
#
#     while cap.isOpened():
#
#     # This will create snapshot of signed letters.
#

    # loop through actions
    # for action in actions:
    #     # loop through sequences aka videos
    #     for sequence in range(no_sequences):
    #         # loop through video length aka sequence length
    #         for frame_num in range(sequence_length):
    #
    #             ret, image = cap.read()  # it is going to read the feed while the camera is opened. It is very fast and its pretty much like a video.
    #
    #             # make detection
    #             image, results = mediapipe_detection(image,holistic)
    #
    #             draw_landmarks(image, results)
    #
    #             draw_styled_landmarks(image,results)
    #
    #
    #
    #
    #             # extract_keypoints(results)[:-10]
    #
    #             # it help Awais to have better understanding of what the program is doing while collecting the fingerspelling data
    #
    #             if frame_num == 0:
    #                 # It is going to inform the programmer that it is starting to collect the data
    #                 cv2.putText(image, 'STARTING COLLECTION', (150,400),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 10, cv2.LINE_AA)
    #
    #                 # It is going to let the programmer know what he should sign
    #                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,50),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    #
    #                 # Show to screen
    #                 cv2.imshow('OpenCV Feed', image)
    #                 cv2.waitKey(2000)  # take a break for 2 seconds before going onto the next letter
    #             else:
    #                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,50),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    #
    #                 cv2.imshow('OpenCV Feed', image)  # it is the name of the image
    #
    #             # saving
    #             keypoints = extract_keypoints(results)
    #             npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
    #             np.save(npy_path,keypoints)
    #
    #             if cv2.waitKey(10) & 0xFF == ord('q'):  # it is going to quit running when the user hit "q" from their keyboard
    #                 break
    # cap.release()
    # cv2.destroyAllWindows()  # close down the image

