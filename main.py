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

# This block of codes below here perform a loop of capturing the frames and then extract them to the mediapipe
cap = cv2.VideoCapture(0)  # accessing the webcam which located at 0

with mp_hands.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, image = cap.read()  # it is going to read the feed while the camera is opened. It is super fast and its pretty much like a video.

        # make detection
        image, results = mediapipe_detection(image,holistic)

        draw_landmarks(image, results)

        # this block of codes below here doesn't work. :/ I am trying to make a different color for the left and
        # right hands. For the left hand, it will be blue and the right hand will be red. Also, I want to make the
        # circle on each keypoint larger.
        #
        draw_styled_landmarks(image,results);


        cv2.imshow('OpenCV Feed', cv2.flip(image, 1))  # it is the name of the image

        if cv2.waitKey(10) & 0xFF == ord('q'):  # it is going to quit running when the user hit "q" from their keyboard
            break
cap.release()
cv2.destroyAllWindows()  # close down the image

