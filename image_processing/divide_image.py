import cv2 as cv
import numpy as np
import sys
from histrograms_calculation import calc_histograms
from block_normalization import normalize_blocks


def calc_new_shape(height, width):
    print "Height: " + str(height)
    print "Width: " + str(width)

    new_height = 128
    new_width = (new_height * width) / height

    if new_width > 64:
        new_height = (64 * new_height) / new_width
        new_width = 64

    print "New height: " + str(new_height)
    print "New width: " + str(new_width)
    return new_height, new_width


def resize_image(image):
    height, width = image.shape
    new_height, new_width = calc_new_shape(height, width)
    resized_image = cv.resize(image, (new_width, new_height), interpolation = cv.INTER_NEAREST)
    final_image = cv.copyMakeBorder(resized_image, 0, 128 - new_height, 0, 64 - new_width, cv.BORDER_CONSTANT)

    return final_image


if __name__ == "__main__":
    inImage = sys.argv[1]
    image = cv.imread(inImage, 0)
    image = np.float32(image) / 255.0

    print "---[ Resizing image... ]---"
    resized_image = resize_image(image)
    print "---[ Image resized ]---\n"

    cv.imshow("Original", image)
    cv.imshow("Resized #1", resized_image)
    cv.waitKey(0)

