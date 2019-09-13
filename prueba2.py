import cv2
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream



class kalma_filter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, x, y):

        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)
        predict = self.kalman.predict()
        return predict



cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("video_luz_natural.mp4")
#cap = cv2.VideoCapture("video_luz_natural_2.mp4")
kernel = np.ones((5,5),np.uint8)

# Buffer to draw tracking path
bufferSize = 64
pts = deque(maxlen=bufferSize)
contador_rojo=0
contador_azul=0
contador_amarillo=0
dim_x = cap.get(3)
kalman = kalma_filter()
predic = []
while(1):
    #print("amarillos= ",contador_amarillo)
    #print("rojos= ",contador_rojo)
    #print("azules= ",contador_azul)

    # Take each frame
    _, frame = cap.read()

    #dim_x, dim_y,_ = frame.shape
    #print ("dim_x",dim_x,"dim_y",dim_y)
    limite=int(dim_x/2)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([105,30,20])
    upper_blue = np.array([135,255,255])

    # define range of red color in HSV
    lower_red = np.array([0,100,45])
    upper_red = np.array([3,255,255])

    lower_red_1 = np.array([177,100,45])
    upper_red_1 = np.array([180,255,255])

    # define range of yellow color in HSV
    lower_yellow = np.array([19,56,54])
    upper_yellow = np.array([29,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # close
    mask_blue=cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
	# Erode 
    mask_blue = cv2.erode(mask_blue, None, iterations=10)
    # open
    mask_blue=cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
	# dilate 
    mask_blue = cv2.dilate(mask_blue, None, iterations=8)


    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

	# Erode 2 times
    mask_yellow = cv2.erode(mask_yellow, None, iterations=8)

	# dilate 2 times
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=6)

	# Erode 2 times
    mask_red = cv2.dilate(mask_red, None, iterations=8)

	# dilate 2 times
    mask_red = cv2.erode(mask_red, None, iterations=10)

	# Erode 2 times
    mask_red_1 = cv2.erode(mask_red_1, None, iterations=8)

	# dilate 2 times
    mask_red_1 = cv2.dilate(mask_red_1, None, iterations=10)

    
    res_blue = cv2.bitwise_and(frame,frame, mask= mask_blue)
    res_red = cv2.bitwise_and(frame,frame, mask= mask_red)
    res_red_1 = cv2.bitwise_and(frame,frame, mask= mask_red_1)
    res_red = cv2.bitwise_or(res_red,res_red_1)
    res_yellow = cv2.bitwise_and(frame,frame, mask= mask_yellow)
    res = cv2.bitwise_or(res_yellow,res_blue)
    res = cv2.bitwise_or(res,res_red)


    

	# find contours in the mask and initialize the current
	# (x, y) center of the ball

	# Get contours



    cnts_red,_= cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts_blue,_= cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    cnts_yellow,_= cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
    Canny=cv2.Canny(res, 70, 250)
    cnts,_= cv2.findContours(Canny.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)

    for i in cnts:
        # calculate moments of binary image
        M = cv2.moments(i)
        

        # calculate x,y coordinate of center
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            cX = 0
            cY = 0

        predic = kalman.Estimate(cX, cY)
        #predic.append((int(tp[0]),int(tp[1])))
        # put text and highlight the center
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.circle(frame, (predic[0], predic[1]), 5, (0, 255, 0), -1)
        #cv2.putText(frame, "centroid", (predic[0] - 25, predic[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #print ("cX=",cX)
        
        if predic[0] <= limite-10:# and cX >= limite+10:
            value_blue = res_blue[int(predic[0]), int(predic[1])]
            value_red = res_red[int(predic[0]), int(predic[1])]
            value_yellow = res_yellow[int(predic[0]), int(predic[1])]
            #print ("valor",value)
            """
            #print ("estoy en el limite")
            if value_red.all() != 0:
                print ("ROJO")
            if value_blue.all() != 0:
                print ("AZUL")
            if value_yellow.all() != 0:
                print ("AMARILLO")
            """


            ad = cv2.bitwise_or(frame, frame, mask= mask_red)

            if ad.all()!= 0:
                print("si")

            #print ("prueba=",res[cX][cY])
            #print ("azul",res)

    
    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print("amarillos= ",contador_amarillo)
print("rojos= ",contador_rojo)
print("azules= ",contador_azul)
