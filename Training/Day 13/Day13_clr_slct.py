# color based ROI detection
# ROI - region of interest Bmin - 71 , Bmax = 142
# Gmin - 94 , Gmax 218
# R min=120 R max = 235

import cv2
import numpy as np
import imutils
h=75;s=100;v=119
lowBGR = np.array([0,98,115])
highBGR =np.array([99,157,255]) 
# copy the path of the video
#path = 'C:\\Users\\pc\\vid_mpeg4.mp4'
path = "VIRAT_S_050201_05_000890_000944.mp4"

# create the video reader object
vid = cv2.VideoCapture(path)
print(vid)
print(vid.isOpened())
c_list = []
while(vid.isOpened()):
    val,frame=vid.read()
    image_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # if the frame is captured
    if(val):
        mask_image = cv2.inRange(image_hsv,
                                (lowBGR),
                                (highBGR))
        # find contours
        cnts = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = imutils.grab_contours(cnts)
        #big object detection
        for c in final_contours:
            area = cv2.contourArea(c)
            if(area>150):
                # print(area)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                c_list.append([cx,cy])
                # c have every info about contour
                cv2.drawContours(frame, [c], -1, (0,0,255))
                cv2.circle(frame, (cx,cy), 4 ,(0,255,0))
                for i in c_list:
                    frame[i[1],i[0]] = (0,0,255) 
    
                cv2.imshow('Frame',frame)
                #cv2.imshow('mask_image',mask_image)
    if(cv2.waitKey(1)==ord('q')):
        break
    

    
vid.release() # close the object
cv2.destroyAllWindows()  