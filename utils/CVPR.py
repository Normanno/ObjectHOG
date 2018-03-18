import os
from xml.etree import ElementTree as ET
import cv2 as cv


class AnnotationParser:
    def __init__(self, path, parse_obects_list=None, roi_width=64, roi_height=128):
        if not os.path.exists(path):
            raise IOError('Error: ' + str(path) + " no such file or directory!")
        self.annotation_path = path
        self.room = str(path).split('.')[0].rsplit('/', 1)[0].rsplit('/', 1)[1]
        self.file_base_name = str(path).split('.')[0].rsplit('/', 1)[1]
        self.objects_list = parse_obects_list
        self.parsed_objects = dict()
        self.parsed_objects_number = dict()
        self.min_width = roi_width
        self.min_height = roi_height
        self.image = None
        self.parse()

    def extract_minimum_bounding_box(self, bds):
        bounding_box = [[bds["min_x"], bds["max_y"]], [bds["max_x"], bds["max_y"]], [bds["max_x"], bds["min_y"]], [bds["min_x"], bds["min_y"]]]

        return bounding_box

    def extract_minimum_bounding_box_pts(self, x_pts, y_pts):
        return self.extract_minimum_bounding_box(self.extract_boundings(x_pts, y_pts))

    def extract_boundings(self, x_pts, y_pts):
        max_x = max(x_pts)
        min_x = min(x_pts)
        max_y = max(y_pts)
        min_y = min(y_pts)
        boundings = {"max_x": max_x, "min_x": min_x, "max_y": max_y, "min_y": min_y}
        return boundings

    def get_file_basename(self):
        return self.file_base_name

    def get_file_room_name(self):
        return self.room

    def parse(self):
        tree = ET.parse(self.annotation_path)
        root = tree.getroot()

        for node in root.findall('object'):
            name = str(node.find('name').text).replace("\n", "")
            if (self.objects_list is not None and name in self.objects_list) or \
                    (self.objects_list is None):
                if name not in self.parsed_objects.keys():
                    self.parsed_objects[name] = list()
                    self.parsed_objects_number[name] = 0
                self.parsed_objects_number[name] += 1
                poly = node.find('polygon')
                x_pts = [int(pt.find('x').text) for pt in poly.findall('pt')]
                y_pts = [int(pt.find('y').text) for pt in poly.findall('pt')]

                # Add the minimum boundings to the parsed objects list
                self.parsed_objects[name].append(self.extract_boundings(x_pts, y_pts))
        img_path = str(self.annotation_path)
        img_path = img_path.replace('Annotations', 'Images').replace('xml', 'jpg')
        self.image = cv.imread(img_path, 0)

    def get_parse_objects_names(self):
        return self.parsed_objects.keys()

    def get_object_rois(self, objectname=None):
        rois = list()
        if objectname is None:
            for n in self.parsed_objects.keys():
                rois.extend(self.parsed_objects[n])
        elif objectname in self.parsed_objects.keys():
            rois = self.parsed_objects[objectname]
        return rois

    def get_parsed_classes(self):
        return self.parsed_objects.keys()

    def get_image_from_roi(self, roi, border=0):
        return self.image[roi["min_y"]-border: roi["max_y"]+border, roi["min_x"]-border: roi["max_x"]+border]

    def get_parsed_object_number_dict(self):
        return self.parsed_objects_number
