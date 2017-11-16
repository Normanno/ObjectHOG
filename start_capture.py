import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)

if capture.isOpened() == False:
    print "Error on opening video streaming"

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        width, height, channels = frame.shape
        print "width: " + str(width)
        print "height: " + str(height)
        print "channels: " + str(channels)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        width, height = frame.shape
        print "width gray: " + str(width)
        print "height gray: " + str(height)
        print "channels gray: " + str(channels)
        for i in range(0, width):
            for j in range(0, height):
                p = frame[i, j]
                frame[i, j] = 255 - (p - 0)

        cv.imshow('Frame', frame)
        if cv.waitKey(3) == 27:
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
