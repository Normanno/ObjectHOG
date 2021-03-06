import sys
import os
from os import listdir
from os.path import exists, splitext
from xml.dom import minidom
from utils.CVPR import AnnotationParser
import math


def parseAnnotation(annotation_file, classes_list):
    xmldoc = minidom.parse(annotation_file)
    root = xmldoc.getElementsByTagName('annotation')
    for node in root[0].childNodes:
        if node.nodeType == node.ELEMENT_NODE:
            if node.nodeName == "object":
                for object in node.childNodes:
                    if object.nodeType == object.ELEMENT_NODE:
                        if object.nodeName == "name":
                            objectname = object.firstChild.nodeValue.strip()
                            if objectname in classes_list:
                                return True
    return False


def parse_annotation_dict(annotation_file, classes_dict):
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
    classes_list = classes_counts.keys()
    for cl in classes_list:
        cl_trainig_file = open(ts_files_dir + '/' + cl + '_training_set.txt', 'w+')
        cl_test_file = open(ts_files_dir + '/' + cl + '_test_set.txt', 'w+')
        ts_number = int(math.ceil((classes_counts[cl] / 100.0) * 70))
        ts_counter = 0
        print "class : " +str(cl)
        for f in classes_files[cl].keys():
            if ts_counter < ts_number:
                cl_trainig_file.write(str(classes_files[cl][f]) + '\t' + f + '\n')
                ts_counter += classes_files[cl][f]
            else:
                cl_test_file.write(str(classes_files[cl][f]) + '\t' + f + '\n')
        cl_trainig_file.close()
        cl_test_file.close()
        print '---[ ' + str(cl) + ' training: ' + str(ts_counter) + ' -- test: ' + str(classes_counts[cl] - ts_counter) + ']---'


def select_negative_ts(classes_list, classes_counts, all_objects_file, annotations_list):
    print "selecting_negative_ts"
    total = sum(n for n in classes_counts.values())     # total-count of rois
    num_of_classes = float(len(classes_list.keys()))    # number of selected classes
    need_items = int(round(total/num_of_classes))       # number of rois to assign to
    negative_ts_annotations = dict()
    other_objects = dict()
    with open(all_objects_file, 'r') as of:
        for line in of:
            key = str(line).replace('\n', '').replace('\t', '')
            other_objects[key] = 0
    print 'need ' + str(need_items) + ' items for negative_ts'
    while need_items > 0 and len(annotations_list) > 0:
        annotation_file = annotations_list.pop()
        handler = AnnotationParser(annotation_file)
        handler.parse()
        parsed_objects_dict = handler.get_parsed_object_number_dict()
        for key in parsed_objects_dict.keys():
            if key in other_objects.keys():
                other_objects[key] += parsed_objects_dict[key]
                need_items -= parsed_objects_dict[key]
                if annotation_file not in negative_ts_annotations.keys():
                    negative_ts_annotations[annotation_file] = 0
                negative_ts_annotations[annotation_file] += parsed_objects_dict[key]
            else:
                print 'key: ' + key + ' not in all objects_keys'

    print 'found ' + str(int(round(total/num_of_classes)) - need_items) + ' for negative_ts'
    return negative_ts_annotations, int(round(total/num_of_classes)) - need_items


def balance_classes(classes_to_balance, classes_dict, classes_counts):
    balanced_classes_dict = dict((x, dict()) for x in classes_dict.keys())
    balanced_classes_counts = dict((x, 0) for x in classes_dict.keys())
    tot = sum([x for x in classes_counts.values()])
    max_for_class = int(round(tot/len(classes_counts.keys())))
    print "Balancing classes : " + str(classes_counts)
    print "Balancing parameters : \n-total: " + str(tot) + "\n-max_for_class: " + str(max_for_class)
    print "Before balancing :\n"+str(classes_counts)
    for cl in classes_to_balance:
        for annotation_file in classes_dict[cl]:
            parser = AnnotationParser(annotation_file, (cl if cl != 'none' else None))
            parser.parse()
            if balanced_classes_counts[cl] >= max_for_class:
                print "done for " + cl
                break
            else:
                add = 0
                print "Parser dict "+str(parser.get_parsed_object_number_dict())
                parsed_object_number = parser.get_parsed_object_number_dict()
                if cl != 'none' and cl in parsed_object_number.keys():
                    add = parser.get_parsed_object_number_dict()[cl]
                elif cl == 'none':
                    add = sum([x for x in parser.get_parsed_object_number_dict().values()])
                if add > 0:
                    balanced_classes_dict[cl][annotation_file] = add
                    balanced_classes_counts[cl] += add
    print "After balancing :\n" + str(balanced_classes_counts)

    return balanced_classes_dict, balanced_classes_counts


def create_files_list(directory):
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
    print "--Found "+str(len(usable_files)) + " annotations --"
    # print files
    print "---[ Usable files list created ]---"
    # Create classes list
    print "---[ Creating classes list... ]---"
    with open(classes_file) as f:
        classes_list = f.readlines()
    classes_dict = dict((x.strip(), dict()) for x in classes_list)
    classes_list = classes_dict.keys()
    print "---[ Classes list created ]---"

    # Search files containing one of the classes
    print "---[ Searching for images containing the objects... ]---"

    unselected_annotations = list() # negative training set selection
    classes_counts = dict((x, 0) for x in classes_list)
    for annotation_file in usable_files:
        parse_dict = dict((x, 0) for x in classes_list)
        if parse_annotation_dict(annotation_file, parse_dict):
            if len(set(classes_list).intersection(parse_dict.keys())) > 0:
                for cl in classes_list:
                    if parse_dict[cl] > 0:
                        classes_dict[cl][annotation_file] = parse_dict[cl]
                        classes_counts[cl] += parse_dict[cl]
        else:
            unselected_annotations.append(annotation_file)

    del usable_files

    print "---[ Search completed ]---\n"
    classes_lists_dir = conf_dir + '/classes_lists'
    if not os.path.isdir(classes_lists_dir):
        os.mkdir(classes_lists_dir)

    print "---[Balancing selected dataset]---"
    classes_to_balance = classes_dict.keys()
    classes_to_balance.remove('none')
    classes_dict, classes_counts = balance_classes(classes_to_balance, classes_dict, classes_counts)
    print "---[Balancing Done!]---"

    if 'none' in classes_list:
        all_objects_file = 'model_processing/CVPR09/all_objects'
        negative_ts_annotations_files, number_of_items = select_negative_ts(classes_dict, classes_counts, all_objects_file, unselected_annotations)
        classes_dict['none'] = negative_ts_annotations_files
        classes_counts['none'] = number_of_items

    print "---[Saving selected dataset info]---"
    save_classes_list(classes_dict, classes_lists_dir, classes_counts)
    print "---[Saving Done!]---"
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
