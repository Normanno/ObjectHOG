import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

bin_position_9 = {
    0: 0,
    20: 1,
    40: 2,
    60: 3,
    80: 4,
    100: 5,
    120: 6,
    140: 7,
    160: 8,
}

bin_position_12 = {
    0: 0,
    30: 1,
    60: 2,
    90: 3,
    120: 4,
    150: 5,
    180: 6,
    210: 7,
    240: 8,
    270: 9,
    300: 10,
    330: 11
}

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

# TODO VECTORIZE THIS IN calc_histograms to have bins
def bin_count(magnitude, angle, cell_dimension, start_point, bin_dimension, signed_angles):
    '''
    :param magnitude: 
    :param angle: 
    :param cell_dimension: 
    :param start_point: array with x,y of the first point of the top-left point of the cell
    :param bin_dimension: number of bins, 9 for signed angles, 12 for unsigned
    :param signed_angles: True if the angles are signed, False otherwise
    :return:
    '''
    bins = np.zeros(shape=bin_dimension)
    for i in range(0, cell_dimension):
        for j in range(0, cell_dimension):
            point_angle = angle[ (start_point[0] +i), (start_point[1] +j)]
            point_magnitude = magnitude[(start_point[0]+i), (start_point[1] +j)]
            #todo verify contributon with angles
            if not signed_angles:
                index = (point_angle - (point_angle % 20)) % 160
                #index = ((point_angle - (point_angle % 20)) / 20) % 16
                position = bin_position_9[index]
                if position != 160:
                    bins[position] += point_magnitude
                else:
                    bins[0] += point_magnitude / 2
                    bins[position] += point_magnitude / 2

            else:
                index = (point_angle - (point_angle % 30)) % 330
                #index = ((point_angle - (point_angle % 30)) / 30) % 33
                position = bin_position_12[index]
                if position != 330:
                    bins[position] += point_magnitude
                else:
                    bins[0] += point_magnitude / 2
                    bins[position] += point_magnitude / 2

    return bins

def calc_histograms(image, magnitude, angle, unsigned=False):
    """
    Cell dimensions: 8x8x1 (1 channel)
    Info in every pixel: 8x8x2 (magnitude and angle)
    Bins number: 9 (best results in paper for signed gradients)
        OR
    Bins number: 12 (best results in paper for unsigned gradients)
    """

    bins_number = 9
    if not unsigned:
        bins_number = 12
    cell_dimension = 8
    image_height, image_width, image_channels = image.shape
    hist_width = image_width / cell_dimension
    hist_height = image_height / cell_dimension
    # creation fo a ndarray (full of zeros) in which the third dimension represents
    # the bins for the cell located by first two
    histograms = np.zeros(shape=(hist_width, hist_height, bins_number))

    for i in range(0, hist_height):
        for j in range(0, hist_width):
            print "CELL: (" + str(i + 1) + ", " + str(j + 1) + ")"
            for k in range(0, cell_dimension):
                for l in range(0, cell_dimension):
                    row = (i * cell_dimension) + k
                    col = (j * cell_dimension) + l
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
