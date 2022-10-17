# importing the libraries and frameworks into this project

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# this block of codes below here detects the hands
mp_hands = mp.solutions.hands  # Hands model
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing Styles
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities






# This block of codes below here perform a loop of capturing the frames and then extract them to the mediapipe
cap = cv2.VideoCapture(0)  # accessing the webcam which located at 0

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, image = cap.read()  # it is going to read the feed while the camera is opened. It is super fast and its pretty much like a video.

        # this block of codes below here converts the color to RGB and then back to BGR
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # this block of codes below here doesn't work. :/ I am trying to make a different color for the left and
        # right hands. For the left hand, it will be blue and the right hand will be red. Also, I want to make the
        # circle on each keypoint larger.

        if results.left_hand_landmarks.landmark & results.right_hand_landmarks.landmark:  # it targets the hands
            for hand_landmarks in results.left_hand_landmarks, results.right_hand_landmarks :
        #         # Draw the hand annotations on the image.
                mp_drawing.hand_landmarks(image,results.left_hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color= (121,22,76), thickness = 2, circle_radius = 4),
                                          mp_drawing.DrawingSpec(color= (121,44,250), thickness = 2, circle_radius = 2)),
                mp_drawing.hand_landmarks(image, results.right_hand_landmarks, mp.hand.HAND_CONNECTIONS,
                                            mp.drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                            mp.drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


        cv2.imshow('OpenCV Feed', cv2.flip(image, 1))  # it is the name of the image

        if cv2.waitKey(10) & 0xFF == ord('q'):  # it is going to quit running when the user hit "q" from their keyboard
            break
cap.release()
cv2.destroyAllWindows()  # close down the image
