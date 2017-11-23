import cv2 as cv
import numpy as np
import multiprocessing as mp
import functools as ft
from matplotlib import pyplot as plt
import sys
import os
from histrograms_calculation import calc_histograms

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

def normalize_block(histograms):
    """
    Normalize the blocks of cells
    :param histograms: a np.matrix containing the histograms of the cells that will form the block
    :return: block
    """

    # Get the size and the number of the histograms
    histogram_rows, histograms_columns, histograms_length = histograms.shape
    histograms_number = histogram_rows * histograms_columns
    print "- Histogram number: " + str(histograms_number)
    print "- Histogram length: " + str(histograms_length)

    # Calculate the size of the block to return and initialize it
    block_length = histograms_length * histograms_number
    print "- Block length: " + str(block_length)
    block = np.zeros(block_length)

    # Fill the block
    index = 0
    for i in range(0, histogram_rows):
        for j in range(0, histograms_columns):
            block[index:index + histograms_length] = histograms[i][j][:]
            index = index + histograms_length

    # Show the block
    # print "- Block: "
    # print block

    # Normalize the block
        # Calculate the L2 norm
    square_block = [np.square(x) for x in block]
    l2_norm = np.sqrt(np.sum(square_block))
    print "- L2 norm: " + str(l2_norm)
        # Normalize the block by dividing every element by the norm
    normalized_block = [x / l2_norm for x in block]

    # Show the normalized block
    # print "- Normalized block: "
    # print normalized_block

    return normalized_block


def build_feature_vector(normalized_blocks):
    """
    Build the HOG feature vector
    :param normalized_blocks: a matrix containing all the normalized blocks
    :return: feature_vector
    """

    # Get the size and the number of the normalized blocks
    blocks_number, blocks_length = normalized_blocks.shape
    print "- Blocks number: " + str(blocks_number)
    print "- Blocks length: " + str(blocks_length)

    # Calculate the size of the feature vector to return and initialize it
    feature_vector_length = blocks_length * blocks_number
    print "- Feature vector length: " + str(feature_vector_length)
    feature_vector = np.zeros(feature_vector_length)

    # Fill the feature vector
    blocks_count = 0
    index = 0
    while index < feature_vector_length:
        feature_vector[index:index + blocks_length] = normalized_blocks[blocks_count][:]
        index = index + blocks_length
        blocks_count = blocks_count + 1

    # Show the feature vector
    # print "- HOG feature vector: "
    # print feature_vector

    return feature_vector


if __name__ == "__main__":
    inImage = sys.argv[1]
    image = cv.imread(inImage, 0)
    image = np.float32(image) / 255.0

    print "---[ Calculating magnitude and angle... ]---"
    magnitude, angle = calc_gradient(image)
    print "---[ Magnitude and angle calculated ]---\n"

    print "---[ Calculating cell histogram... ]---"
    histograms = calc_histograms(image, magnitude, angle, True)
    print "---[ Cell histogram calculated ]---\n"

    print "---[ Normalizing blocks... ]---"
    histograms_rows, histograms_columns, histograms_length = histograms.shape
    histograms_number = histograms_rows * histograms_columns
    histograms_per_block = 4
    histograms_per_block_sqrt = np.int_(np.sqrt(histograms_per_block))

    if histograms_per_block_sqrt <= histograms_rows and histograms_per_block_sqrt <= histograms_columns:
        print "---[ Normalizing blocks... ]---"
        # Calculate blocks number and length
        blocks_number = (histograms_rows - histograms_per_block_sqrt + 1) * (histograms_columns - histograms_per_block_sqrt + 1)
        blocks_length = histograms_length * histograms_per_block

        # Normalize blocks
        normalized_blocks = np.zeros([blocks_number, blocks_length])
        blocks_count = 0
        for i in range(0, histograms_rows - histograms_per_block_sqrt + 1):
            for j in range(0, histograms_columns - histograms_per_block_sqrt + 1):
                print "Block #" + str(blocks_count + 1)
                normalized_blocks[blocks_count][:] = normalize_block(histograms[i:i + histograms_per_block_sqrt, j:j + histograms_per_block_sqrt])
                blocks_count = blocks_count + 1
                print ""
        print "---[ Blocks normalized ]---\n"

        print "---[ Building feature vector... ]---"
        feature_vector = build_feature_vector(normalized_blocks)
        print "---[ Feature vector built ]---\n"
