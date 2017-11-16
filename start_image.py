import cv2 as cv
import sys
import os
from matplotlib import pyplot as plt
from image_processing.gamma_correction import gamma_correction
from image_processing.histogram import extract_grey_levels_histogram
import numpy as np


def imageProcess(src, dest):
    if not os.path.exists(src):
        print "Error: " + src + " No such file or directory!!!!"
        return
    if os.path.exists(dest):
        print "Warning: " + dest + " File already exists!!!!"

    srcImg = cv.imread(src)
    histogram = extract_grey_levels_histogram(srcImg)
    destImg = gamma_correction(srcImg, 1/1.0, 1)

    while True:
        cv.imshow("src", srcImg)
        cv.imshow("dest", destImg)
        plt.plot(histogram)
        if cv.waitKey(5) == 27:
            break

    cv.destroyAllWindows()
    cv.imwrite(dest, destImg)


if __name__ == "__main__":
    inImage = sys.argv[1]
    outImage = sys.argv[2]
    imageProcess(inImage, outImage)
