import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


def extract_grey_levels_histogram(src):
    histogram = cv.calcHist([src], [0], None, [256], [0, 256])
    return histogram


if __name__ == '__main__':
    src_path = sys.argv[1]
    dest_path = sys.argv[2]

    if not os.path.exists(src_path):
        print "Error: " + src_path + " No such file or directory!!!!"
        exit(1)
    if os.path.exists(dest_path):
        print "Warning: " + dest_path + " File already exists!!!!"
        exit(2)
    srcImg = cv.imread(src_path)

    histogram = extract_grey_levels_histogram(srcImg)
    cv.imshow("histogram", histogram)

    cv.imwrite(dest_path, histogram)
