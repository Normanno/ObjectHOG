import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


def calc_gradient(image):
    gX = cv.Sobel(image, cv.CV_32F, 1,0, 1)
    gY = cv.Sobel(image, cv.CV_32F, 0, 1, 1)
    magnitude, angle = cv.cartToPolar(gX, gY, True)

    cv.imshow("Magnitude", magnitude)
    cv.imshow("Angle", angle)
    cv.imshow("GX", gX)
    cv.imshow("GY", gY)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return magnitude, angle


def calc_histogram(cell):
    histogram = 0
    return histogram


def normalize_block(block):
    block = 0
    return block


def build_feature_vector(image):
    feature_vector = 0
    return feature_vector


if __name__ == "__main__":
    inImage = sys.argv[1]
    image = cv.imread(inImage, 0)
    image = np.float32(image) / 255.0

    print "---[ Calculating magnitude and angle... ]---"
    magnitude, orientation = calc_gradient(image)
    print "---[ Magnitude and angle calculated ]---"

    print "---[ Calculating cell histogram... ]---"
    histogram = calc_histogram(image)
    print "---[ Cell histogram calculated ]---"

    print "---[ Normalizing block... ]---"
    block = normalize_block(image)
    print "---[ Block normalized ]---"

    print "---[ Building feature vector... ]---"
    feature_vector = build_feature_vector(image)
    print "---[ Feature vector built ]---"
