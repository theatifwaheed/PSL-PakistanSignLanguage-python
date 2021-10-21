import cv2 as cv
import numpy as np

cap = cv.VideoCapture('1depth.avi')

while True:
    _, frame = cap.read()

    kernel = np.ones((2, 2), np.uint8)

    # cv.imshow ('frame', frame)

    imGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    imGray = imGray.astype(np.uint8)

    print (imGray.max(), imGray.min())

    th = cv.inRange(imGray, 60, 80)

    # _, mask = cv.threshold(imgray, 1, 255, cv.THRESH_BINARY)
    # closing... erode to dilate
    # Convo. neural network...

    erosion = cv.erode(th, kernel, iterations=1)

    dilation = cv.dilate(erosion, kernel, iterations=1)

    Contours, Hierarchy = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    MaxC = max(Contours, key=cv.contourArea)

    for contour in Contours:
        hull = cv.convexHull(MaxC)

    final = cv.drawContours(frame, [hull], -1, (0, 0, 255), 2)

    print("Contours " + str(len(MaxC)))

    cv.imshow('Threshold ', dilation)

    cv.imshow('Erosion ', erosion)

    cv.imshow('Hull ', final)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()