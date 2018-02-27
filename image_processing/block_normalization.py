import cv2 as cv
import numpy as np
import multiprocessing as mp
import functools as ft
import time
from matplotlib import pyplot as plt
import sys
import os


class NormalizeBlocksHolder:

    def __init__(self, shape):
        self.normalized_blocks = np.zeros(shape=shape)

    def set_block(self, block_index, new_block):
        self.normalized_blocks[block_index][:] = new_block

    def set_block_single(self, res):
        self.set_block(res[0], res[1])

    def get_blocks(self):
        return self.normalized_blocks


def normalize_block(block_index, histograms):
    """
    Normalize the blocks of cells
    :param block_index: index of the normalized block, used for block holder
    :param histograms: a np.matrix containing the histograms of the cells that will form the block
    :return: block
    """

    # Get the size and the number of the histograms
    histogram_rows, histograms_columns, histograms_length = histograms.shape
    histograms_number = histogram_rows * histograms_columns
    #print "- Histogram number: " + str(histograms_number)
    #print "- Histogram length: " + str(histograms_length)

    # Calculate the size of the block to return and initialize it
    block_length = histograms_length * histograms_number
    #print "- Block length: " + str(block_length)
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
    #print "- L2 norm: " + str(l2_norm)
        # Normalize the block by dividing every element by the norm
    normalized_block = [x / l2_norm for x in block]

    # Show the normalized block
    # print "- Normalized block: "
    # print normalized_block
    return block_index, normalized_block


def normalize_blocks(histograms, stamp=False):
    histograms_rows, histograms_columns, histograms_length = histograms.shape
    histograms_number = histograms_rows * histograms_columns
    histograms_per_block = 4
    histograms_per_block_sqrt = np.int_(np.sqrt(histograms_per_block))

    normalized_blocks_holder = None

    if histograms_per_block_sqrt <= histograms_rows and histograms_per_block_sqrt <= histograms_columns:
        if stamp:
            print "---[ Normalizing blocks... ]---"
        # Calculate blocks number and length
        blocks_number = (histograms_rows - histograms_per_block_sqrt + 1) * (
        histograms_columns - histograms_per_block_sqrt + 1)
        blocks_length = histograms_length * histograms_per_block

        normalized_blocks_holder = NormalizeBlocksHolder((blocks_number, blocks_length))

        # Normalize blocks
        #normalized_blocks = np.zeros([blocks_number, blocks_length])
        blocks_count = 0
        process_pool = mp.Pool(processes=mp.cpu_count() / 2)
        start_time = time.time()
        hist_rows = histograms_rows - histograms_per_block_sqrt + 1
        hist_columns = histograms_columns - histograms_per_block_sqrt + 1
        for i in range(0, hist_rows):
            for j in range(0, hist_columns):
                #print "Block #" + str(blocks_count + 1)
                process_pool.apply_async(ft.partial(normalize_block,  blocks_count, histograms[i:i + histograms_per_block_sqrt, j:j + histograms_per_block_sqrt]
                                                    ), range(hist_rows, hist_columns),
                                         callback=normalized_blocks_holder.set_block_single)
                #normalized_blocks[blocks_count][:] = normalize_block(
                #    histograms[i:i + histograms_per_block_sqrt, j:j + histograms_per_block_sqrt])
                blocks_count = blocks_count + 1
                #print ""

        process_pool.close()
        process_pool.join()
        elapsed = time.time() - start_time
        if stamp:
            print "*********milliseconds elapsed " + str(elapsed)

    return None if normalized_blocks_holder is None else normalized_blocks_holder.get_blocks()