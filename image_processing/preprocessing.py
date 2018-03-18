import cv2 as cv
import numpy as np
import math
import sys
import os
import warnings

def automatic_gamma_correction_gray(src):
    # Bit depth of the image
    img_range = 1
    half_range = 1
    if src.dtype == 'uint8':
       img_range = 8
       half_range = 4
    elif src.dtype == 'uint16':
       img_range = 16
       half_range = 8
    avg_color = src[:, :].mean()
    gamma = math.log(half_range*img_range, 10) / math.log(avg_color/img_range, 10)
    warnings.filterwarnings('error')
    try:
        res = (img_range + half_range) * np.power(src/img_range, 1/(gamma if gamma != 0 else 1))
    except RuntimeError, e:
        print " message "+ e.message
        print "( "+str(img_range)+" + "+str(half_range)+" ) * np.power("+str(src)+"/"+str(img_range)+", 1/"+str(gamma)+")"
        print "avg_color " + str(avg_color)

    return res


def preprocess(img, gamma):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if gamma:
        img = automatic_gamma_correction_gray(img)
    return img

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

    destImg = automatic_gamma_correction_gray(srcImg)

    cv.imwrite(dest_path, destImg)

