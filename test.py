import cv2
from cvzone import FPS
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import time
import mediapipe as mp
import imutils
from imutils.video import FPS


class Video:

    def __init__(self):
        self.frame = None

    def handle(self):
        # This block of codes below here perform a loop of capturing the frames and then extract them to the mediapipe

            try:
                offset = 20
                imgSize = 300
                detector = HandDetector(maxHands=1)
                classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
                labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                          "T", "U", "V", "W", "X", "Y", "Z"]
                cap = cv2.VideoCapture(0)  # accessing the webcam which located at 0
                time.sleep(1)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                fps = FPS().start()

                while (cap.isOpened()):  # made change here

                    ret, image = cap.read()  # it is going to read the feed while the camera is opened. It is super fast and its pretty much like a video.

                    imgOutput = image.copy()
                    hands, image = detector.findHands(image)
                    if hands:
                        hand = hands[0]
                        x, y, w, h = hand['bbox']

                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgCrop = image[y - offset:y + h + offset, x - offset:x + w + offset]



                        aspectRatio = h / w

                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)
                            # print(prediction, index)

                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)

                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                                  (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255),
                                2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset),
                                  (x + w + offset, y + h + offset), (255, 0, 255), 4)

                    retention, jpeg = cv2.imencode('.jpeg', imgOutput)

                    if (retention):
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                        fps.update()

                    fps.stop()
                    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


            except(Exception):
                cap.release()
                cv2.destroyAllWindows()
                print('error producing frame')

    def build_frame(image):
        ret, jpeg = cv2.imencode('.jpg', image)
        if (ret):
            yield jpeg.tobytes()

    def kill(self):
        # self.cap.release()
        cv2.destroyAllWindows()  # close down the image