import cv2 as cv
import numpy as np
import sys
import os


def negative(src, lambda_value, constant_value ):
    width, heigh, channel = src.shape
    dest = src.clone()
    for i in range(0, width):
        for j in range(0, heigh):
            dest[i, j] = 255 - (src[i, j] - 0)

    return dest

if __name__ == "__main__":
    src_path = sys.argv[1]
    dest_path = sys.argv[2]

    if not os.path.exists(src_path):
        print "Error: " + src_path + " No such file or directory!!!!"
        exit(1)
    if os.path.exists(dest_path):
        print "Warning: " + dest_path + " File already exists!!!!"
        exit(2)
    srcImg = cv.imread(src_path)

    lambda_value = 0.5
    constant_value = 1.0
    if len(sys.argv) > 3:
        lamda_value = float(sys.argv[3])
    if len(sys.argv) > 4:
        constant_value = float(sys.argv[4])
    destImg = negative(srcImg, lambda_value, constant_value)

    cv.imwrite(dest_path, destImg)
