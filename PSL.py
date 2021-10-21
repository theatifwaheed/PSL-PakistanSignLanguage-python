import cv2 as cv
import numpy as np

cap = cv.VideoCapture('1depth.avi')

while True:
    _, frame = cap.read()
    kernel = np.ones((2, 2), np.uint8)
    #cv.imshow('frame', frame)
    imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgray = imgray.astype(np.uint8)
    print (imgray.max(), imgray.min())
    th = cv.inRange(imgray, 60, 80)
    #_, mask = cv.threshold(imgray, 1, 255, cv.THRESH_BINARY)
    # closing... erode to dilate
    #convo. neural network...
    erosion = cv.erode(th, kernel, iterations=1)

    dilation = cv.dilate(erosion, kernel, iterations=1)
    countours, heirarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    MaxC = max(countours, key=cv.contourArea)
    for countour in countours:
        hull = cv.convexHull(MaxC)
    final = cv.drawContours(frame, [hull], -1, (0, 0, 255), 2)
    print("countours: " + str(len(MaxC)))
    cv.imshow('threshold', dilation)
    cv.imshow('erotion', erosion)
    cv.imshow('Hull', final)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()