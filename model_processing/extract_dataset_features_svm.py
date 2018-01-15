import sys
import os
import numpy as np
from image_processing.feature_extraction import feature_extraction
from model_processing.utils.CVPR import AnnotationParser


def extract_training_features_SVM(classes_training_files, feature_dir, roi_width=64, roi_height=128):
    for cl in classes_training_files.keys():
        for fi in classes_training_files[cl]:
            for tr_file in fi.keys():
                annotation_parser = AnnotationParser(tr_file, [cl])
                for roi in annotation_parser.get_object_rois(cl):
                    image = annotation_parser.get_image_from_roi(roi)
                    features = feature_extraction(image)


def load_model_config(model_dir):
    """
    >load_model_config(model_dir)
    :param model_dir: root of model directory
    :return: dict(class, dict(annotation_file_path, #of_class_objects_in_annotation)
    """

    if model_dir[len(model_dir)-1:] != '/':
        model_dir += '/'
    if not os.path.exists(model_dir):
        raise IOError("ERROR: "+model_dir+" no such file or directory")

    classes_path = model_dir + 'classes.txt'

    classes_training_files_path = model_dir + 'training_data/'

    classes = []
    classes_training_files_lists_dict = dict()

    with open(classes_path, 'r') as classes_file:
        for line in classes_file:
            c = line.replace('\n', '').strip()
            classes.append(c)
            classes_training_files_lists_dict[c] = dict()

    for key in classes_training_files_lists_dict.keys():
        class_training_files_list_path = classes_training_files_path + key + "_training_set.txt"
        if os.path.exists(class_training_files_list_path):
            with open(class_training_files_list_path, 'r') as tsf_list:
                for line in tsf_list:
                    line = line.replace('\n', '').strip()
                    items = line.split('\t')
                    classes_training_files_lists_dict[items[1]] = int(items[0])
        else:
            print 'Error: ' + class_training_files_list_path + ' no such file or directory'

    return classes_training_files_lists_dict

if __name__ == '__main__':
    print' dataset features extraction '
    if len(sys.argv) < 2:
        print '***** Error *****'
        print 'This program requires 1 arguments ('+str(len(sys.argv)-1)+') given :'
        print '(1) model src directory, (root of the model config dir)'
        print '(2)(optional) roi width (default 64)'
        print '(3)(optional) roi height (default 128)'
        exit(1)

    model_dir = sys.argv[1]
    classes_dict = None
    if model_dir[len(model_dir) - 1:] != '/':
        model_dir += '/'

    roi_width = 64
    roi_height = 128
    if len(sys.argv) > 2:
        roi_width = int(sys.argv[2])

    if len(sys.argv) > 3:
        roi_height = int(sys.argv[3])

    if not os.path.isdir(model_dir):
        print '***** Error: ' + model_dir + ' does not exist!'
        exit(1)
    try:
        classes_dict = load_model_config(model_dir)
    except IOError as ex:
        print 'Error during config files load \n\t\t - ' + str(ex)
        exit(1)

    if classes_dict is None:
        print "Error: classes not loaded!"
        exit(1)

    training_data_feat_dir = model_dir + 'training_data_feats_w'+str(roi_width)+'_h'+str(roi_height)+'/'
    if not os.path.exists(training_data_feat_dir):
        os.makedirs(training_data_feat_dir)

    extract_training_features_SVM(classes_dict, training_data_feat_dir, roi_width, roi_height)
