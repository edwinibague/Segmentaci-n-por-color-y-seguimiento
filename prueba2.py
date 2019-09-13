import cv2
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video_luz_natural.mp4")
kernel = np.ones((5,5),np.uint8)

# Buffer to draw tracking path
bufferSize = 64
pts = deque(maxlen=bufferSize)


while(1):

    # Take each frame
    _, frame = cap.read()

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
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

	# Erode 2 times
    mask_blue = cv2.erode(mask_blue, None, iterations=2)

	# dilate 2 times
    mask_blue = cv2.dilate(mask_blue, None, iterations=2)

	# Erode 2 times
    mask_yellow = cv2.erode(mask_yellow, None, iterations=2)

	# dilate 2 times
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=2)

	# Erode 2 times
    mask_red = cv2.erode(mask_red, None, iterations=2)

	# dilate 2 times
    mask_red = cv2.dilate(mask_red, None, iterations=2)

	# Erode 2 times
    mask_red_1 = cv2.erode(mask_red_1, None, iterations=2)

	# dilate 2 times
    mask_red_1 = cv2.dilate(mask_red_1, None, iterations=2)

    
    res_blue = cv2.bitwise_and(frame,frame, mask= mask_blue)
    res_red = cv2.bitwise_and(frame,frame, mask= mask_red)
    res_red_1 = cv2.bitwise_and(frame,frame, mask= mask_red_1)
    res_red = cv2.bitwise_or(res_red,res_red_1)
    res_yellow = cv2.bitwise_and(frame,frame, mask= mask_yellow)


    

	# find contours in the mask and initialize the current
	# (x, y) center of the ball

	# Get contours



    cnts,_= cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
	#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    print(cnts)

	# Check if contours were found
    if len(cnts) > 0:
        # Find the biggest area contour
        c = max(cnts, key=cv2.contourArea)
        # Extract the circle that encloses the contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Add the current center to the queue
    pts.appendleft(center)
    
    # Loop over the queue
    
    for i in range(1, len(pts)):
        # Ignore null points
        if pts[i - 1] is None or pts[i] is None:
            continue
            # Draw the line
        thickness = int(np.sqrt(bufferSize / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)




    # Aplicar las respectivas mascaras y obtener una combinacion de las mascaras resultantes
    res = cv2.bitwise_and(frame,frame, mask= mask_blue)
    res = cv2.Canny(res,	180,	200)


    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()