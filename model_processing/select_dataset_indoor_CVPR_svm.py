import sys
import os
from os import listdir
from os.path import exists, splitext
from xml.dom import minidom
import math

def parseAnnotation(annotation_file, classes_list):
    """
    > parseAnnotation(annotation_file, classes_list)
    Parses the input annotation to search for the objects that have to be recognized

    :param annotation_file: the annotation to be parsed
    :param classes_list: the list containing the objects to be recognized
    :return: a boolean value indicating if the annotation contains one of the objects or not
    """
    xmldoc = minidom.parse(annotation_file)
    root = xmldoc.getElementsByTagName('annotation')
    # annotation = {'filename': "NULL", 'folder': "NULL", 'objectscount': 0, 'objects': []}
    for node in root[0].childNodes:
        if node.nodeType == node.ELEMENT_NODE:
            if node.nodeName == "filename":
                filename = node.firstChild.nodeValue.strip()
                # annotation['filename'] = filename
            # elif node.nodeName == "folder":
            #     folder = node.firstChild.nodeValue.strip()
            #     annotation['folder'] = folder
            # elif node.nodeName == "object":
            elif node.nodeName == "object":
                # annotation['objectscount'] = annotation['objectscount'] + 1
                # objectinfo = {'name': "NULL", 'id': "NULL", 'polygon': {'pointscount': 0, 'points': []}}
                for object in node.childNodes:
                    if object.nodeType == object.ELEMENT_NODE:
                        if object.nodeName == "name":
                            objectname = object.firstChild.nodeValue.strip()
                            # objectinfo['name'] = objectname
                            if objectname in classes_list:
                                # print "Found in: '" + filename + "'"
                                return True
                        # elif object.nodeName == "id":
                        #     objectid = object.firstChild.nodeValue.strip()
                        #     objectinfo['id'] = objectid
                #         elif object.nodeName == "polygon":
                #             for point in object.childNodes:
                #                 if point.nodeType == point.ELEMENT_NODE:
                #                     if point.nodeName == "pt":
                #                         objectinfo['polygon']['pointscount'] = objectinfo['polygon']['pointscount'] + 1
                #                         pointcoordinates = []
                #                         for coordinate in point.childNodes:
                #                             if coordinate.nodeType == coordinate.ELEMENT_NODE:
                #                                 if coordinate.nodeName == "x":
                #                                     pointcoordinates.append(coordinate.firstChild.nodeValue.strip())
                #                                 elif coordinate.nodeName == "y":
                #                                     pointcoordinates.append(coordinate.firstChild.nodeValue.strip())
                #                         objectinfo['polygon']['points'].append(pointcoordinates)
                # annotation['objects'].append(objectinfo)
    return False


def parse_annotation_dict(annotation_file, classes_dict):
    """
    > parseAnnotation(annotation_file, classes_list)
    Parses the input annotation to search for the objects that have to be recognized

    :param annotation_file: the annotation to be parsed
    :param classes_dict: the dict containing the objects to be recognized and relative counters
    :return: a boolean value indicating if the annotation contains one of the objects or not
    """
    xmldoc = minidom.parse(annotation_file)
    root = xmldoc.getElementsByTagName('annotation')
    keys = classes_dict.keys()
    result = False

    for node in root[0].childNodes:
        if node.nodeType == node.ELEMENT_NODE:
            if node.nodeName == "filename":
                filename = node.firstChild.nodeValue.strip()
            elif node.nodeName == "object":
                for object in node.childNodes:
                    if object.nodeType == object.ELEMENT_NODE:
                        if object.nodeName == "name":
                            objectname = object.firstChild.nodeValue.strip()
                            if objectname in keys:
                                classes_dict[objectname] += 1
                                result = True
    return result


def save_classes_list(classes_dict, classes_list_dir, classes_counts):
    """
    > save_classes_list(classes_dict, classes_list_dir)
    Save for each class the list of files ( with the relative quantity) containing object of the class
    and a file containing the total number of objects for each class

    :param classes_dict: dictionary of classes , each element is a dictionary of files path with an
    int representing the quantity
    :param classes_list_dir: the directory where the list files have to be written
    :param classes_counts: dict of classes with the number of objects found, for each class, in the dataset
    :return:
    """
    if not os.path.exists(classes_list_dir):
        os.mkdir(classes_list_dir)

    for cl in classes_dict.keys():
        list_file = open(classes_list_dir + '/' + cl + '.txt', 'w+')
        for item in classes_dict[cl]:
            list_file.write(str(classes_dict[cl][item]) + "\t" + item + "\n")
        list_file.close()

    cl_counts = open(classes_list_dir + '/classes_counts.txt', 'w+')
    for cl in classes_counts.keys():
        cl_counts.write(cl + '\t' + str(classes_counts[cl]) + '\n')
    cl_counts.close()


def ts_selection(classes_counts, classes_files, ts_files_dir):
    """
    >ts_selection(classes_counts, classes_files, ts_files_dir)
    Splits the file in training (70%) and test set (30%) for each class, then save these lists in ts_files_dir

    :param classes_counts: dictionary of classes with the relative amount of instances in the files
    :param classes_files: dictionary of classes , each element is a dictionary of files path with an
    int representing the quantity
    :param ts_files_dir: directory where the configuration files has to be written
    :return:
    """

    classes_list = classes_counts.keys()
    for cl in classes_list:
        cl_trainig_file = open(ts_files_dir + '/' + cl + '_training_set.txt', 'w+')
        cl_test_file = open(ts_files_dir + '/' + cl + '_test_set.txt', 'w+')
        ts_number = int(math.ceil((classes_counts[cl] / 100.0) * 70))
        ts_counter = 0
        for f in classes_files[cl].keys():
            if ts_counter < ts_number:
                cl_trainig_file.write(str(classes_files[cl][f]) + '\t' + f + '\n')
                ts_counter += classes_files[cl][f]
            else:
                cl_test_file.write(str(classes_files[cl][f]) + '\t' + f + '\n')
        cl_trainig_file.close()
        cl_test_file.close()
        print '---[ ' + str(cl) + ' training: ' + str(ts_counter) + ' -- test: ' + str(classes_counts[cl] - ts_counter) + ']---'


def create_files_list(directory):
    """
    > createFilesList(directory)
    Create a list of annotation files that are physically associated with an image file (otherwise, it's impossible
    to verify the presence or not of the objects)

    :param directory: the source directory
    :return: a list of files that are physically associated with an image
    """
    files = []
    for subdir in listdir(directory + "/Images/"):
        for image in listdir(directory + "/Images/" + subdir):
            name = splitext(image)[0]
            if exists(directory + "/Annotations/" + subdir + "/" + name + ".xml"):
                files.append(directory + "/Annotations/" + subdir + "/" + name + ".xml")
    return files


def select(src_dir, conf_dir, classes_file):
    # Create usable files list
    print "---[ Creating usable files list... ]---"
    usable_files = create_files_list(src_dir)
    # print files
    print "---[ Usable files list created ]---\n"

    # Create classes list
    print "---[ Creating classes list... ]---"
    with open(classes_file) as f:
        classes_list = f.readlines()
    #classes_list = [x.strip() for x in classes_list]
    classes_dict = dict((x.strip(), dict()) for x in classes_list)
    classes_list = classes_dict.keys()
    # print content
    print "---[ Classes list created ]---\n"

    # Search files containing one of the classes
    print "---[ Searching for images containing the objects... ]---"
    # dict(class, dict(file, #ofObjectOfTypeClass))
    #containing = dict((x, dict()) for x in classes_list)
    classes_counts = dict((x, 0) for x in classes_list)
    for annotation_file in usable_files:
        parse_dict = dict((x, 0) for x in classes_list)
        if parse_annotation_dict(annotation_file, parse_dict):
            for cl in classes_list:
                if parse_dict[cl] > 0:
                    classes_dict[cl][annotation_file] = parse_dict[cl]
                    classes_counts[cl] += parse_dict[cl]
    # print containing
    print "---[ Search completed ]---\n"
    classes_lists_dir = conf_dir + '/classes_lists'
    if not os.path.isdir(classes_lists_dir):
        os.mkdir(classes_lists_dir)
    save_classes_list(classes_dict, classes_lists_dir, classes_counts)
    #print "---[ " + str(classes_dict) + " ]---\n"
    print "---[ Classes files lists saved in " + classes_lists_dir + " ]---"
    training_data_dir = conf_dir + '/training_data'
    if not os.path.isdir(training_data_dir):
        os.mkdir(training_data_dir)
    ts_selection(classes_counts, classes_dict, training_data_dir)
    print "---[ Training data configuration file saved in " + training_data_dir + " ]---"


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print '***** Error *****'
        print 'This program requires 3 arguments ('+str(len(sys.argv))+') given :'
        print '(1) src directory, root of the dataset)'
        print '(2) conf directory, where to put lists of files for positive-ts,' \
              'negative-ts, test-set, validation-set'
        print '(3) classes file, list of object to classify'
        exit(1)

    src_dir = sys.argv[1]
    conf_dir = sys.argv[2]
    classes_file = sys.argv[3]
    if not os.path.isdir(src_dir):
        print '***** Error: ' + src_dir + ' does not exist!'
        exit(1)

    if not os.path.isdir(conf_dir):
        print '***** Error: ' + conf_dir + ' does not exist!'
        exit(1)

    if not os.path.isfile(classes_file):
        print '***** Error: ' + classes_file + ' does not exist!'
        exit(1)

    select(src_dir, conf_dir, classes_file)
