import cv2 as cv
import numpy as np
import sys
from divide_image import resize_image
from histrograms_calculation import calc_histograms
from block_normalization import normalize_blocks
from preprocessing import preprocess


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

    return feature_vector


def feature_extraction(image, unsigned=True, stamp=False, gamma=True):
    feature_vector = None
    image = preprocess(image, gamma)
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

    normalized_blocks = normalize_blocks(histograms)
    if normalized_blocks is not None:
        print "---[ Building feature vector... ]---"
        feature_vector = build_feature_vector(normalized_blocks)
        print "---[ Feature vector built ]---\n"

