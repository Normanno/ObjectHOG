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
    gradient_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=1)
    gradient_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    cv.imshow("X gradient", gradient_x)
    cv.imshow("Y gradient", gradient_y)
    cv.imshow("Magnitude", magnitude)
    cv.imshow("Angle", angle)
    cv.waitKey(0)
    cv.destroyAllWindows()

    angle = np.uint32(angle)

    return magnitude, angle


# TODO vectorize this in calc_histograms to have bins
def bin_count(magnitude, angle, cell_dimension, start_point, bin_dimension, signed_angles):
    """
    :param magnitude:
    :param angle:
    :param cell_dimension:
    :param start_point: array with x,y of the first point of the top-left point of the cell
    :param bin_dimension: number of bins, 9 for signed angles, 12 for unsigned
    :param signed_angles: True if the angles are signed, False otherwise
    :return:
    """

    bins = np.zeros(shape=bin_dimension)
    for i in range(0, cell_dimension):
        for j in range(0, cell_dimension):
            point_angle = angle[(start_point[0] + i), (start_point[1] + j)]
            point_magnitude = magnitude[(start_point[0] + i), (start_point[1] + j)]
            # TODO verify contribution with angles
            # DONE
            bin_step = 30
            bin_max = 360
            if not signed_angles:
                bin_step = 20
                bin_max = 180
            # contribution is the percentage of contribution to the first bin after the one located by position
            contribution = ((point_angle % bin_step) * (100/bin_step)) / 100
            index = (point_angle - (point_angle % bin_step)) % bin_max

            if not signed_angles:
                position = bin_position_9[index]
                position_next = bin_position_9[(index + bin_step) % bin_max]
            else:
                position = bin_position_12[index]
                position_next = bin_position_12[(index + bin_step) % bin_max]

            if contribution == 0.0:
                bins[position] += point_magnitude
            else:
                bins[position] += point_magnitude * (1 - contribution)
                bins[position_next] += point_magnitude * contribution

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
    image_height, image_width = image.shape
    hist_width = image_width / cell_dimension
    hist_height = image_height / cell_dimension
    # creation fo a ndarray (full of zeros) in which the third dimension represents
    # the bins for the cell located by first two
    histograms = np.zeros(shape=(hist_width, hist_height, bins_number))

    for i in range(0, hist_height):
        for j in range(0, hist_width):
            # print "CELL: (" + str(i + 1) + ", " + str(j + 1) + ")"
            for k in range(0, cell_dimension):
                for l in range(0, cell_dimension):
                    row = (i * cell_dimension) + k
                    col = (j * cell_dimension) + l
                    # print str(row) + ", " + str(col)
                    if angle[row][col] > 180:
                        angle[row][col] = angle[row][col] - 180

    return histograms


def normalize_block(histograms):
    """
    :param histograms: a np.matrix containing the histograms of the cells that will form the block
    :return: block
    """

    # Get the size and the number of the histograms
    histograms_number, histograms_length = histograms.shape
    print "- Histogram number: " + str(histograms_number)
    print "- Histogram length: " + str(histograms_length)

    # Calculate the size of the block to return and initialize the block
    block_length = histograms_length * histograms_number
    print "- Block length: " + str(block_length)
    block = np.zeros(block_length)

    # Fill the block
    histograms_count = 0
    index = 0
    while index < block_length:
        block[index:index + histograms_length] = histograms[histograms_count][:]
        index = index + histograms_length
        histograms_count = histograms_count + 1

    # Normalize the block
        # Calculate the L2 norm
    l2_norm = 0
    for i in range(0, block_length):
        l2_norm = l2_norm + np.square(block[i])
    l2_norm = np.sqrt(l2_norm)
    print "- L2 norm: " + str(l2_norm)
        # Normalize the block by dividing every element by the norm
    normalized_block = np.zeros(block_length)
    for i in range(0, block_length):
        normalized_block[i] = np.divide(block[i], l2_norm)

    # Show the normalized block
    # print "- Normalized block: "
    # print normalized_block

    return normalized_block


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
    block = normalize_block(image[0:2, 0:16])
    print "---[ Block normalized ]---\n"

    print "---[ Building feature vector... ]---"
    feature_vector = build_feature_vector(image)
    print "---[ Feature vector built ]---\n"
