import cv2 as cv
import numpy as np
import multiprocessing as mp
import functools as ft
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


def bin_count(magnitude, angle, bins, start_point, cell_dimension, signed_angles):
    """
    :param magnitude:
    :param angle:
    :param bins:
    :param cell_dimension:
    :param start_point: array with x,y of the first point of the top-left point of the cell
    :param signed_angles: True if the angles are signed, False otherwise
    :return:
    """
    for i in range(0, cell_dimension):
        for j in range(0, cell_dimension):
            point_angle = angle[(start_point[0] + i), (start_point[1] + j)]
            point_magnitude = magnitude[(start_point[0] + i), (start_point[1] + j)]
            bin_step = 30
            bin_max = 360
            if not signed_angles:
                bin_step = 20
                bin_max = 180
            # contribution is the percentage of contribution to the first bin after the one located by position
            # e.g. point_angle=155 bin_step=20
            # contributions = (( 155%20 ) * (100/20)) /100 = (15 * 5) /100 = 75/100 = 0.75
            # the point contributes for 75% to the bin representing 160 (because is closer) and
            # for 25% to thebin representig 140
            contribution = ((point_angle % bin_step) * (100/bin_step)) / 100
            # index is the index in the bin_position_* map , used to find the bin index in the new map
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
        print str(start_point) + str(bins)
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
    hist_rows = image_height / cell_dimension
    hist_columns = image_width / cell_dimension
    # creation fo a ndarray (full of zeros) in which the third dimension represents
    # the bins for the cell located by first two
    histograms = np.zeros([hist_rows, hist_columns, bins_number])

    for i in range(0, hist_rows):
        for j in range(0, hist_columns):
            # print "CELL: (" + str(i + 1) + ", " + str(j + 1) + ")"
            for k in range(0, cell_dimension):
                for l in range(0, cell_dimension):
                    row = (i * cell_dimension) + k
                    col = (j * cell_dimension) + l
                    # print str(row) + ", " + str(col)
                    if angle[row][col] > 180:
                        angle[row][col] = angle[row][col] - 180
                    histograms[i][j] = np.random.randint(100, size=(bins_number))

    # process_pool = mp.Pool(processes=mp.cpu_count()/2)
    # for i in range(0, hist_width):
    #     for j in range(0, hist_height):
    #         res = process_pool.imap(ft.partial(bin_count, magnitude, angle, histograms[i, j], [i, j],
    #                                      cell_dimension, not unsigned), range(hist_width * hist_height))
    #         print str(res.get())
    # async_pool = mp.Pool(processes=mp.cpu_count()/2)
    # m_res = []
    #
    # #process_pool.close()
    # #process_pool.join()

    return histograms


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

    # Calculate blocks number and length
    blocks_number = (histograms_rows - 1) * (histograms_columns - 1)
    blocks_length = histograms_length * histograms_per_block

    # Normalize blocks
    normalized_blocks = np.zeros([blocks_number, blocks_length])
    blocks_count = 0
    for i in range(0, histograms_rows - 1):
        for j in range(0, histograms_columns - 1):
            print "Block #" + str(blocks_count + 1)
            normalized_blocks[blocks_count][:] = normalize_block(histograms[i:i + 2, j:j + 2])
            blocks_count = blocks_count + 1
            print ""
    print "---[ Blocks normalized ]---\n"

    print "---[ Building feature vector... ]---"
    feature_vector = build_feature_vector(normalized_blocks)
    print "---[ Feature vector built ]---\n"
