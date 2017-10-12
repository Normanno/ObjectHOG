import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

if capture.isOpened() == False:
    print "Error on opening video streaming"

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        cv.imshow('Frame', frame)
        if cv.waitKey(3) == 27:
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
