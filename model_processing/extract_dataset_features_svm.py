import os
import sys
import numpy as np
from image_processing.feature_extraction import feature_extraction
from utils.CVPR import AnnotationParser
from image_processing.divide_image import resize_image


def extract_features_svm(classes_data_files, feature_dir, model_dir, roi_width=64, roi_height=128, generate_labels=False):
    """
    >extract_features_svm(classes_training_files, feature_dir, roi_width=64, roi_height=128)
     Extract for each object and each annotation file, the features of the object's instances within the file,
     then saves the features array in the feature_dir.
     For each annotation file - object type a .npz file will be created.
    :param classes_data_files: dict(key:object, value:dict( key:annotation file , value: #of object contained))
    :param feature_dir: folder to save files
    :param model_dir: model folder
    :param roi_width: equal to the model width
    :param roi_height: equal the model height
    :return:
    """
    classes_labels = {}
    label_counter = 0
    for cl in classes_data_files.keys():
        class_feat_dir = os.path.join(feature_dir, cl)
        if not os.path.exists(class_feat_dir):
            os.makedirs(class_feat_dir)
        for fi in classes_data_files[cl]:
            print "class : " + cl + " - file : " + fi
            cls = None
            if cl != 'none':
                cls = [cl]
            annotation_parser = AnnotationParser(fi, cls)
            features_file_path = os.path.join(class_feat_dir, annotation_parser.get_file_basename())
            features_arrays = list()
            for roi in annotation_parser.get_object_rois(cl if cl != 'none' else None):
                try:
                    image = annotation_parser.get_image_from_roi(roi)
                    image = resize_image(image, model_width=roi_width, model_height=roi_height)
                    features_arrays.append(feature_extraction(image))
                except Exception:
                    print 'Exception: \n-class ' + cl + '\n-Annotation: ' + fi + '\n-ROI: ' + str(roi)

            save_features_to_npz(features_arrays, features_file_path)
        classes_labels[label_counter] = cl
        label_counter += 1
    if generate_labels:
        with open(os.path.join(model_dir, 'classes_labels.txt'), 'w+') as classes_labels_file:
            for key in sorted(classes_labels.keys()):
                classes_labels_file.write(str(classes_labels[key]) + "\t" + str(key) + "\n")


def save_features_to_npz(feat_array, feat_file_path):
    """
    >save_features_npz(feat_array, feat_file)
     Write the feat_array to feat_file (npz format see numpy.savez for more info)
    :param feat_array:
    :param feat_file_path:
    :return:
    """
    if os.path.exists(feat_file_path):
        raise IOError('Error: ' + feat_file_path + ' already exists!')

    np.savez(feat_file_path, feat_array)


def load_features_from_npz(npz_file_path):
    """
    >load_features_from_npz(npz_file_path)
     Reads all the features contained in the file specified by npz_file_path and returns a list of ndarray
    :param npz_file_path:
    :return:
    """
    try:
        npz_file = np.load(npz_file_path)
    except IOError as ioe:
        print 'Error : '+str(ioe)
        exit(1)
    feats = list()
    for f in npz_file.files:
        feats.extend(npz_file[f])
    return feats


def load_data_annotations(model_dir, ends_with):
    """
    >load_data_annotations(model_dir)
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

    #Read selected classes
    with open(classes_path, 'r') as classes_file:
        for line in classes_file:
            c = line.replace('\n', '').strip()
            classes.append(c)
            classes_training_files_lists_dict[c] = dict()

    #Read annotation files locations for each class
    for key in classes_training_files_lists_dict.keys():
        class_training_files_list_path = classes_training_files_path + key + ends_with
        if os.path.exists(class_training_files_list_path):
            with open(class_training_files_list_path, 'r') as tsf_list:
                for line in tsf_list:
                    line = line.replace('\n', '').strip()
                    items = line.split('\t')
                    classes_training_files_lists_dict[key][items[1]] = int(items[0])
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
    classes_test_dict = None
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
        classes_dict = load_data_annotations(model_dir, "_training_set.txt")
        classes_test_dict = load_data_annotations(model_dir, "_test_set.txt")
    except IOError as ex:
        print 'Error during config files load \n\t\t - ' + str(ex)
        exit(1)

    if classes_dict is None:
        print "Error: classes not loaded!"
        exit(1)

    training_data_feat_dir = model_dir + 'training_data_feats_w'+str(roi_width)+'_h'+str(roi_height)+'/'
    if not os.path.exists(training_data_feat_dir):
        os.makedirs(training_data_feat_dir)

    testing_data_feat_dir = model_dir + 'testing_data_feats_w'+str(roi_width)+'_h'+str(roi_height)+'/'
    if not os.path.exists(testing_data_feat_dir):
        os.makedirs(testing_data_feat_dir)

    extract_features_svm(classes_dict, training_data_feat_dir, model_dir, roi_width, roi_height, True)
    extract_features_svm(classes_test_dict, testing_data_feat_dir, model_dir, roi_width, roi_height)

