from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils

# Green limits in HSV
greenLower = (100, 40, 20)
greenUpper = (140, 255, 255)
# Buffer to draw tracking path
bufferSize = 64
pts = deque(maxlen=bufferSize)
# Object to read camera, 0 means first webcam
vs = VideoStream(src=0).start()

while True:
	# Wait for a frame
	frame = vs.read()
	frame = cv2.flip( frame, 1 )

	# Resize the image
	frame = imutils.resize(frame, width=600)
	# Filter image
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	# Convert to HSV color space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Threshold image
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	# Erode 2 times
	mask = cv2.erode(mask, None, iterations=20)
	# dilate 2 times
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball

	# Get contours
	cnts,_= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
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