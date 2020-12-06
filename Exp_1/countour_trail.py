import cv2
import argparse
import imutils
import numpy as np
from collections import deque

pts = deque(maxlen = 64)
capture = cv2.VideoCapture("1.avi")
pos_file = open("pos_txt.txt","w")

if capture.isOpened() is False:
    print("error opening the video file")

while capture.isOpened():
    ret,frame = capture.read()

    if ret is True:

        cv2.imshow('Original frame',frame)
        #b_frame = cv2.GaussianBlur(frame,(11,11),0)
        b_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,thr = cv2.threshold(b_frame,45,255,cv2.THRESH_BINARY_INV)
        
        thr = cv2.erode(thr,None, iterations= 2)
        thr = cv2.dilate(thr, None,iterations= 2)
        cv2.imshow("thresholded",thr)

        cnts = cv2.findContours(thr.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if(len(cnts)>0):
            c = max(cnts, key= cv2.contourArea)
            ((x,y),radius) = cv2.minEnclosingCircle(c)
            M =cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            st = "{} {} \n".format(center[0],center[1])
            pos_file.write(st)

            if radius > 10:
                cv2.circle(frame,(int(x),int(y)),int(radius),(0,25,255),2)
                #cv2.circle(frame,center,5,(100,0,255),-1)

        pts.appendleft(center)
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue

            thickness = int(np.sqrt(64/float(i+1)) * 2.5)
            cv2.line(frame, pts[i-1],pts[i],(0,100,250),thickness)

        
        cv2.imshow('Updated frame',frame)



        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    else:
        break
pos_file.close()
capture.release()
cv2.destroyAllWindows()