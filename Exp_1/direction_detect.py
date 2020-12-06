import cv2
import argparse
import imutils
import numpy as np
from collections import deque


pos_list = []
pts = deque(maxlen = 64)
capture = cv2.VideoCapture("4.avi")


counter = 0
(dX, dY) = (0, 0)
direction = ""


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
            pos_list.append(center)

            if radius > 10:
                cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
                cv2.circle(frame,center,5,(100,0,255),-1)
                pts.appendleft(center)

        for i in np.arange(1, len(pts)):

            if pts[i - 1] is None or pts[i] is None:
                continue

            if counter >= 10 and i == 10 and pts[i-10] is not None:
                dX = pts[i-10][0] - pts[i][0]
                dY = pts[i-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")

                if np.abs(dX) > 10:
                    dirX = "RIGHT" if np.sign(dX) == 1 else "LEFT"

                if np.abs(dY) > 10:
                    dirY = "UP" if np.sign(dY) == 1 else "DOWN"

                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

                else:
                    direction = dirX if dirX != "" else dirY

            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)
        
        cv2.imshow('Updated frame',frame)

        key = cv2.waitKey(1) & 0xFF

        counter += 1

        if key == ord("q"):
            break

    else:
        break

print(pos_list)
capture.release()
cv2.destroyAllWindows()