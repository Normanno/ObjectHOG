import os
import cv2 as cv
import numpy as np
import math
from xml.etree import ElementTree as ET


class Handler:

    def __init__(self, annotation_file):
        print 'parsing'
        self.annotation_file = annotation_file
        self.class_objects = dict()
        self.image_file = ''
        self.parse_annotation()

    def parse_annotation(self):
        tree = ET.parse(self.annotation_file)
        root = tree.getroot()
        self.image_file = str(root.find('.//folder').text.strip()) + "/" + str(root.find('.//filename').text.strip())
        for node in root.findall('.//object'):
            pts = [(int(pt.find('x').text.strip()), int(pt.find('y').text.strip())) for pt in node.findall('.//pt')]
            node_name = node.find('name').text.strip()
            if node_name not in self.class_objects.keys():
                self.class_objects[node_name] = list()
            self.class_objects[node_name].append(pts)

    def get_objects_list(self):
        return self.class_objects.keys()

    def get_polygons(self, class_name):
        if class_name not in self.class_objects.keys():
            return None
        return self.class_objects[class_name]

    def get_image_relative_path(self):
        return self.image_file


def get_minimum_bounding_box(pts_list, square=False):
    """
    > get_minimum_bounding_box(self, pts_list)
    give a polygon represented by pts_list, returns the minimum bounding box as a list of four points
    :param pts_list: list of polygon points
    :param square: if true force the bounding box to be a square
    :return: list of minimum bounding box points
    """
    tl_x = None
    tl_y = None

    br_x = None
    br_y = None

    for pt in pts_list:
        if tl_x is None or tl_y is None or br_x is None or br_y is None:
            tl_x = pt[0]
            tl_y = pt[1]
            br_x = pt[0]
            br_y = pt[1]
        tl_x = min(pt[0], tl_x)
        tl_y = min(pt[1], tl_y)
        br_x = max(pt[0], br_x)
        br_y = max(pt[1], br_y)

    if square:
        side = max(abs(br_x - tl_x), abs(br_y - tl_y))

    return [(tl_x, tl_y), (br_x, tl_y), (br_x, br_y), (tl_x, br_y)]

