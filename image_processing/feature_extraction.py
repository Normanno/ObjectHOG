import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os


def calc_gradient(image):
    gradientX = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=1)
    gradientY = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv.cartToPolar(gradientX, gradientY, angleInDegrees=True)

    cv.imshow("X gradient", gradientX)
    cv.imshow("Y gradient", gradientY)
    cv.imshow("Magnitude", magnitude)
    cv.imshow("Angle", angle)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # PROBLEMA (?): così i valori di angle vanno da 0 a 360, non a 180 come dice nel paper
    # Normalizziamo a 180?
    # angle = np.uint32(angle)
    # print angle

    return magnitude, angle


def calc_histogram(cell):
    """
    Cell dimensions: 8x8x1 (1 channel)
    Info in every pixel: 8x8x2 (magnitude and angle)
    """
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
