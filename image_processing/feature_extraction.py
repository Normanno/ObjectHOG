import cv2 as cv
import numpy as np
import sys
from divide_image import resize_image
from histrograms_calculation import calc_histograms
from block_normalization import normalize_blocks


def calc_gradient(image, stamp=False):
    gradient_x = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=1)
    gradient_y = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

    if stamp:
        cv.imshow("X gradient", gradient_x)
        cv.imshow("Y gradient", gradient_y)
        cv.imshow("Magnitude", magnitude)
        cv.imshow("Angle", angle)
        cv.waitKey(0)
        cv.destroyAllWindows()

    angle = np.uint32(angle)

    return magnitude, angle


def build_feature_vector(normalized_blocks, stamp=False):
    """
    Build the HOG feature vector
    :param normalized_blocks: a matrix containing all the normalized blocks
    :return: feature_vector
    """

    # Get the size and the number of the normalized blocks
    blocks_number, blocks_length = normalized_blocks.shape
    if stamp:
        print "- Blocks number: " + str(blocks_number)
        print "- Blocks length: " + str(blocks_length)

    # Calculate the size of the feature vector to return and initialize it
    feature_vector_length = blocks_length * blocks_number
    if stamp:
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


def feature_extraction(image, unsigned=True, stamp=False):
    """
    >feature_extraction(image)
    this function extracts the feature from the nd
    :param image: float32 ndarray representing the Region Of Interest (ROI)
    :param unsigned: boolean indicate if the bins are signed(False) or unsigned(True), Default is True
    :return: array of features
    """
    feature_vector = None
    image = np.float32(image) / 255.0
    magnitude, angle = calc_gradient(image, stamp=stamp)
    histograms = calc_histograms(image, magnitude, angle, unsigned, stamp=stamp)
    normalized_blocks = normalize_blocks(histograms, stamp=stamp)
    if normalized_blocks is not None:
        feature_vector = build_feature_vector(normalized_blocks, stamp=stamp)
    return feature_vector


if __name__ == "__main__":
    inImage = sys.argv[1]
    image = cv.imread(inImage, 0)
    image = np.float32(image) / 255.0

    resized_image = resize_image(image)

    print "---[ Calculating magnitude and angle... ]---"
    magnitude, angle = calc_gradient(resized_image)
    print "---[ Magnitude and angle calculated ]---\n"

    print "---[ Calculating cell histogram... ]---"
    histograms = calc_histograms(resized_image, magnitude, angle, True)
    print "---[ Cell histogram calculated ]---\n"

    '''
    histograms_rows, histograms_columns, histograms_length = histograms.shape
    histograms_number = histograms_rows * histograms_columns
    histograms_per_block = 4
    histograms_per_block_sqrt = np.int_(np.sqrt(histograms_per_block))

    if histograms_per_block_sqrt <= histograms_rows and histograms_per_block_sqrt <= histograms_columns:
        printdef image_resize() "---[ Normalizing blocks... ]---"
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
    '''
    normalized_blocks = normalize_blocks(histograms)
    if normalized_blocks is not None:

        print "---[ Building feature vector... ]---"
        feature_vector = build_feature_vector(normalized_blocks)
        print "---[ Feature vector built ]---\n"

