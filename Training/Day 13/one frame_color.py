import cv2

path = 'VIRAT_S_050201_05_000890_000944.mp4'

vid = cv2.VideoCapture(path)

#function for color selector with mouse
def mouseRGB(event, x,y,flag,param):
    if (event==cv2.EVENT_FLAG_LBUTTON):
        colorB = frame[x,y,0]
        colorG = frame[x,y,1]
        colorR = frame[x,y,2]
        print('BGR values: ',colorB,colorG,colorR)
        print('corr: ',x,y)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame',mouseRGB)

ret,frame = vid.read()
cv2.imshow('Frame',frame)

while (1):
    if (cv2.waitKey(1) & 0xFF==ord('q')):
        break

vid.release()
cv2.destroyAllWindows()
