# importing the libraries and frameworks into this project

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

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
     'X', 'Y', 'Z'])

# 30 videos worth of data
no_sequences = 30

# the video is going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# This block of codes below here perform a loop of capturing the frames and then extract them to the mediapipe
cap = cv2.VideoCapture(0)  # accessing the webcam which located at 0

with mp_hands.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    # while cap.isOpened():

    # This will create snapshot of signed letters.

    # loop through actions
    for action in actions:
        # loop through sequences aka videos
        for sequence in range(no_sequences):
            # loop through video length aka sequence length
            for frame_num in range(sequence_length):

                ret, image = cap.read()  # it is going to read the feed while the camera is opened. It is very fast and its pretty much like a video.

                # make detection
                image, results = mediapipe_detection(image,holistic)

                draw_landmarks(image, results)

                draw_styled_landmarks(image,results)

                # extract_keypoints(results)[:-10]

                # it help Awais to have better understanding of what the program is doing while collecting the fingerspelling data

                if frame_num == 0:
                    # It is going to inform the programmer that it is starting to collect the data
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)

                    # It is going to let the programmer know what he should sign
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.waitKey(10000) # take a break for 10 seconds before going onto the next letter
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', cv2.flip(image, 1))  # it is the name of the image

                # saving
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path,keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):  # it is going to quit running when the user hit "q" from their keyboard
                    break
cap.release()
cv2.destroyAllWindows()  # close down the image

