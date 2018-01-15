import cv2 as cv
import numpy as np
import math
import sys
import os


def automatic_gamma_correction(src):
    # Bit depth of the image
    img_range = 0
    #TODO use opencv constants
    if src.dtype == 'uint8':
        img_range = 8
    elif src.dtype == 'uint16':
        img_range = 16

    avg_color = src[:, :, 1].mean()
    gamma = math.log(img_range/2*img_range, 10) / math.log(avg_color/img_range, 10)

    return np.power(src, 1/gamma)


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

    destImg = automatic_gamma_correction(srcImg)

    cv.imwrite(dest_path, destImg)

