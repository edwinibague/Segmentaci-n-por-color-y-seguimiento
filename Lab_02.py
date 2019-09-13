import cv2
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)


while(1):

    # Take each frame
    _, frame = cap.read()

    # Filter image
    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([105,30,30])
    upper_blue = np.array([135,255,255])

    # define range of red color in HSV
    lower_red = np.array([0,80,72])
    upper_red = np.array([3,255,255])
    lower_red_1 = np.array([177,80,72])
    upper_red_1 = np.array([180,255,255])

    # define range of yellow color in HSV
    lower_yellow = np.array([20,57,50])
    upper_yellow = np.array([29,255,255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

	# Erode 2 times
    mask_blue = cv2.erode(mask_blue, None, iterations=4)

	# dilate 2 times
    mask_blue = cv2.dilate(mask_blue, None, iterations=4)

    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

	# Erode 2 times
    mask_yellow = cv2.erode(mask_yellow, None, iterations=4)

	# dilate 2 times
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=4)

    # Threshold the HSV image to get only red colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

	# Erode 2 times
    mask_red = cv2.erode(mask_red, None, iterations=4)

	# dilate 2 times
    mask_red = cv2.dilate(mask_red, None, iterations=4)

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

	# Erode 2 times
    mask_red_1 = cv2.erode(mask_red_1, None, iterations=4)

	# dilate 2 times
    mask_red_1 = cv2.dilate(mask_red_1, None, iterations=4)

    # Aplicar las respectivas mascaras y obtener una combinacion de las mascaras resultantes
    res_blue = cv2.bitwise_and(frame,frame, mask= mask_blue)
    res_red = cv2.bitwise_and(frame,frame, mask= mask_red)
    res_red_1 = cv2.bitwise_and(frame,frame, mask= mask_red_1)
    res_red = cv2.bitwise_or(res_red,res_red_1)
    res_yellow = cv2.bitwise_and(frame,frame, mask= mask_yellow)
    res = cv2.bitwise_or(res_yellow,res_blue)
    res = cv2.bitwise_or(res,res_red)

    res = cv2.Canny(res,	180,	200)


    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()