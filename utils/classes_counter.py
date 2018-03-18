from CVPR import AnnotationParser
import sys
from os import listdir
import os

if __name__ == '__main__':
    src = '/run/media/norman/My Norman/UNI/VisioneComputazionale/indoorCVPR_09'
    objects = dict()
    for subdir in listdir(src + "/Annotations"):
        for annotation_file in listdir(src + "/Annotations/" + subdir):
            annotation = AnnotationParser(src + "/Annotations/" + subdir+ "/" +annotation_file)
            annotation.parse()
            parsed_objects = annotation.get_parsed_object_number_dict()
            for key in parsed_objects:
                if key not in objects.keys():
                    objects[key] = 0
                objects[key] += parsed_objects[key]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with file(dir_path + '/objects_numbers', 'w+') as out_file:
        keys = objects.keys()
        keys.sort()
        for key in keys:
            out_file.write(key+"\t"+str(objects[key]))
