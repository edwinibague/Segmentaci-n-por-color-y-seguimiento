import cv2
import numpy as np
from collections import deque
from imutils.video import VideoStream

class kalma_filter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    def correccion(self,x,y,rad):

        med = np.array([[np.float32(x)],[np.float32(y)],[np.float32(rad)]])

        self.kalman.correct(med)
        predict = self.kalman.predict()
        return predict


class Pingpong:
    def __init__(self, color, radio, pos, frame):
        self.Color = color
        self.Radio = radio
        self.Pos = pos
        self.KalmanOb = kalma_filter()
        self.circle = self.circle(x, y, pos, frame)
        self.orden = []

    def circle(self, x, y, radio, frame):

        cv2.circle(frame, (int(x), int(y)), int(radio), (0, 255, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    def new_kalma(self):
        Kalma = self.KalmanOb()