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

    def Estimate(self, x, y):

        med = np.array([[np.float32(x)],[np.float32(y)]])
        self.kalman.correct(med)
        predict = self.kalman.predict()
        return predict


class Pingpong:
    def __init__(self, color, radio, pos, frame):
        self.Color = color
        self.Radio = radio
        self.Pos = pos
        self.KalmanOb = kalma_filter()
        self.orden = []

    def new_kalma(self):
        Kalma = self.KalmanOb()

    def DetectPP(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        Dilated = cv2.dilate(mask, kernel)

        [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(Dilated, 8, cv2.CV_32S)
        stats = np.delete(stats, (0), axis=0)
        try:
            maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)

        # This is our ball coords that needs to be tracked
            ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2]/2)
            ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3]/2)
            return [ballX, ballY]
        except:
               pass

        return [0,0]
