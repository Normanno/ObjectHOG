import sys
import os
from os import listdir
from os.path import exists, splitext
from xml.dom import minidom


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


def createFilesList(directory):
    """
    > createFilesList(directory)
    Create a list of annotation files that are physically associated with an image file (otherwise, it's impossible
    to verify the presence or not of the objects)

    :param directory: the source directory
    :return: a list of files that are physically associated with an image
    """
    files = []
    for subdir in listdir(directory + "/images/"):
        for image in listdir(directory + "/images/" + subdir):
            name = splitext(image)[0]
            if exists(directory + "/annotations/" + subdir + "/" + name + ".xml"):
                files.append(directory + "/annotations/" + subdir + "/" + name + ".xml")
    return files


def select(src_dir, conf_dir, classes_file):
    # Create usable files list
    print "---[ Creating usable files list... ]---"
    usable_files = createFilesList(src_dir)
    # print files
    print "---[ Usable files list created ]---\n"

    # Create classes list
    print "---[ Creating classes list... ]---"
    with open(classes_file) as f:
        classes_list = f.readlines()
    classes_list = [x.strip() for x in classes_list]
    # print content
    print "---[ Classes list created ]---\n"

    # Search files containing one of the classes
    print "---[ Searching for images containing the objects... ]---"
    containing = []
    for annotation_file in usable_files:
        if parseAnnotation(annotation_file, classes_list):
            containing.append(annotation_file)
    # print containing
    print "---[ Search completed ]---\n"


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print '***** Error *****'
        print 'This program requires 3 arguments ('+str(len(sys.argv))+') given :'
        print '(1) src directory, root of the dataset)'
        print '(2) conf directory, where to put lists of files for positive-ts,' \
              'negative-ts, test-set, validation-set'
        print '(3) classes file, list of object ot classify'
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
