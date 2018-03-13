import cv2 as cv
import numpy as np
import sys
from histrograms_calculation import calc_histograms
from block_normalization import normalize_blocks


def calc_new_shape(height, width, model_width=64, model_height=128, stamp=False):
    new_height = model_height
    new_width = (new_height * width) / height

    if new_width > model_width:
        new_height = (model_width * new_height) / new_width
        new_width = model_width

    if stamp:
        print "Height: " + str(height)
        print "Width: " + str(width)
        print "New height: " + str(new_height)
        print "New width: " + str(new_width)

    return new_height, new_width


def resize_image(image, model_width=64, model_height=128):
    height, width = image.shape
    new_height, new_width = calc_new_shape(height, width, model_width, model_height)
    resized_image = cv.resize(image, (new_width, new_height))
    v_correction = 0
    h_correction = 0
    if (model_height - new_height) % 2 == 1:
        v_correction = 1
    if (model_width - new_width) % 2 == 1:
        h_correction = 1
    top_border = (model_height - new_height) / 2
    right_border = ((model_width - new_width) / 2) + h_correction
    bottom_border = ((model_height - new_height) / 2) + v_correction
    left_border = (model_width - new_width) / 2
    final_image = cv.copyMakeBorder(resized_image, top_border, bottom_border, left_border, right_border, cv.BORDER_CONSTANT)
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

