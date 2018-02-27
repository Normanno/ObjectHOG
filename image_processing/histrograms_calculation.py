import cv2 as cv
import numpy as np
import multiprocessing as mp
import functools as ft
import time

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


class HistogramHolder:

    def __init__(self, shape):
        # creation fo a ndarray (full of zeros) in which the third dimension represents
        # the bins for the cell located by first two
        self.histogram = np.zeros(shape=shape)

    def sum_bins(self, x, y, bins):
        self.histogram[x, y] += bins

    def sum_bins_single(self, res):
        self.sum_bins([res[0]], [res[1]], res[2])

    def get_histograms(self):
        return self.histogram


def bin_count(magnitude, angle, start_point, cell_dimension, bins_dimension, signed_angles):
    """
    :param magnitude:
    :param angle:
    :param bins_dimension:
    :param cell_dimension:
    :param start_point: array with x,y of the first point of the top-left point of the cell
    :param signed_angles: True if the angles are signed, False otherwise
    :return:
    """
    bins = np.zeros(shape=bins_dimension)
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
    return start_point[0], start_point[1], bins


def calc_histograms(image, magnitude, angle, unsigned=False, stamp=False):
    """
    Cell dimensions: 8x8x1 (1 channel)
    Info in every pixel: 8x8x2 (magnitude and angle)
    Bins number: 9 (best results in paper for signed gradients)
        OR
    Bins number: 12 (best results in paper for unsigned gradients)
    """
    start_time = time.time()
    bins_number = 9
    if not unsigned:
        bins_number = 12
    cell_dimension = 8
    image_height, image_width = image.shape
    hist_rows = image_height / cell_dimension
    hist_columns = image_width / cell_dimension

    histograms = HistogramHolder((hist_rows, hist_columns, bins_number))

    if unsigned:
        for i in range(0, hist_rows):
            for j in range(0, hist_columns):
                for k in range(0, cell_dimension):
                    for l in range(0, cell_dimension):
                        row = (i * cell_dimension) + k
                        col = (j * cell_dimension) + l
                        if angle[row][col] > 180:
                            angle[row][col] = angle[row][col] - 180

    process_pool = mp.Pool(processes=mp.cpu_count()/2)
    for i in range(0, hist_rows):
        for j in range(0, hist_columns):
            process_pool.apply_async(ft.partial(bin_count, magnitude, angle, [i, j], cell_dimension,
                                                bins_number, not unsigned), range(hist_rows, hist_columns),
                                     callback=histograms.sum_bins_single)
    process_pool.close()
    process_pool.join()
    elapsed = time.time() - start_time
    if stamp:
        print "*********milliseconds elapsed "+str(elapsed)
    return histograms.get_histograms()

