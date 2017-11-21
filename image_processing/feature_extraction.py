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

    angle = np.uint32(angle)

    return magnitude, angle


def calc_histograms(image, magnitude, angle, unsigned=False):
    """
    Cell dimensions: 8x8x1 (1 channel)
    Info in every pixel: 8x8x2 (magnitude and angle)
    Bins number: 12 (best results in paper for angles from 0 to 360)
    """
    bins_number = 12
    image_width = 64
    image_height = 128
    # histograms = np.matrix((image_height / 8), (image_width / 8))
    histograms = 1
    for i in range(0, 16):
        for j in range(0, 8):
            print "CELL: (" + str(i + 1) + ", " + str(j + 1) + ")"
            histogram = np.array(bins_number)
            for k in range(0, 8):
                for l in range(0, 8):
                    row = (i * 8) + k
                    col = (j * 8) + l
                    print str(row) + ", " + str(col)
                    if angle[row][col] > 180:
                        angle[row][col] = angle[row][col] - 180
    print angle

    return histograms


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
    magnitude, angle = calc_gradient(image)
    print "---[ Magnitude and angle calculated ]---\n"

    print "---[ Calculating cell histogram... ]---"
    histograms = calc_histograms(image, magnitude, angle)
    print "---[ Cell histogram calculated ]---\n"

    print "---[ Normalizing block... ]---"
    block = normalize_block(image)
    print "---[ Block normalized ]---\n"

    print "---[ Building feature vector... ]---"
    feature_vector = build_feature_vector(image)
    print "---[ Feature vector built ]---\n"
