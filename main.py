from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils

# Green limits in HSV
blueLower = (100, 40, 20)
blueUpper = (140, 255, 255)

redLower = (170, 90, 85)
redUpper = (179, 255, 255)

yellowLower = (10, 90, 85)
yellowUpper = (20, 255, 255)
# Buffer to draw tracking path
bufferSize = 64
pts = deque(maxlen=bufferSize)
pts2 = deque(maxlen=bufferSize)
pts3 = deque(maxlen=bufferSize)
# Object to read camera, 0 means first webcam
vs = VideoStream(src=0).start()

xmax = 500
ymax = 500

red = 0
blue = 0
yellow = 0

while True:
    # Wait for a frame
    frame = vs.read()
    frame = cv2.flip(frame, 1)

    # Resize the image
    frame = imutils.resize(frame, width=600)
    # Filter image
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Threshold image
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask2 = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask3 = cv2.inRange(hsv, blueLower, blueUpper)

    # mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    # Erode 2 times
    mask = cv2.erode(mask, None, iterations=20)
    mask2 = cv2.erode(mask2, None, iterations = 20)
    mask3 = cv2.erode(mask3, None, iterations = 20)
    # dilate 2 times
    mask = cv2.dilate(mask, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)
    mask3 = cv2.dilate(mask3, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball

    # Get contours
    cnts_red, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_yellow, _ = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts3, _ = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    center2 = None
    center3 = None

    print(cnts_red)

    # Check if contours were found
    if len(cnts_red) > 0:
        # Find the biggest area contour
        c = max(cnts_red, key=cv2.contourArea)
        # Extract the circle that encloses the contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        # only proceed if the radius meets a minimum size
        if x >= xmax:
            red += 1
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Check if contours were found
    if len(cnts2) > 0:
        # Find the biggest area contour
        c2 = max(cnts2, key=cv2.contourArea)
        # Extract the circle that encloses the contour
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        center2 = (int(x2), int(y2))
        # only proceed if the radius meets a minimum size
        if radius2 > 10:
            cv2.circle(frame, (int(x2), int(y2)), int(radius2), (0, 255, 255), 2)
            cv2.circle(frame, (int(x2), int(y2)), 5, (0, 0, 255), -1)

    # Check if contours were found
    if len(cnts3) > 0:
        # Find the biggest area contour
        c3 = max(cnts3, key=cv2.contourArea)
        # Extract the circle that encloses the contour
        ((x3, y3), radius3) = cv2.minEnclosingCircle(c3)
        center3 = (int(x3), int(y3))
        # only proceed if the radius meets a minimum size
        if radius3 > 10:
            cv2.circle(frame, (int(x3), int(y3)), int(radius3), (0, 255, 255), 2)
            cv2.circle(frame, (int(x3), int(y3)), 5, (0, 0, 255), -1)

    # Add the current center to the queue
    pts.appendleft(center)
    pts2.appendleft(center2)
    pts3.appendleft(center3)

    # Loop over the queue
    for i in range(1, len(pts)):
        # Ignore null points
        if pts[i - 1] is None or pts[i] is None:
            continue
        # Draw the line
        thickness = int(np.sqrt(bufferSize / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    for i in range(1, len(pts2)):
        # Ignore null points
        if pts2[i - 1] is None or pts2[i] is None:
            continue
        # Draw the line
        thickness2 = int(np.sqrt(bufferSize / float(i + 1)) * 2.5)
        cv2.line(frame, pts2[i - 1], pts2[i], (0, 0, 255), thickness2)

    for i in range(1, len(pts3)):
        # Ignore null points
        if pts3[i - 1] is None or pts3[i] is None:
            continue
        # Draw the line
        thickness3 = int(np.sqrt(bufferSize / float(i + 1)) * 2.5)
        cv2.line(frame, pts3[i - 1], pts3[i], (0, 0, 255), thickness3)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# stop the camera
vs.stop()

# close all windows
cv2.destroyAllWindows()