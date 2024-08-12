import cv2
import numpy as np

path = 'vid_mpeg4.mp4'

vid = cv2.VideoCapture(path)
print(vid)

while (vid.isOpened()):
    ret,frame = vid.read()
    
    if (ret):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cv2.imshow('RGB',rgb_frame)
    cv2.imshow('GRAY',gray_frame)
    cv2.imshow('HSV',hsv_frame)
    cv2.imshow('YCRCB',ycrcb_frame)
    if (cv2.waitKey(1)==ord('q')):
        break

vid.release()
cv2.destroyAllWinodws()
