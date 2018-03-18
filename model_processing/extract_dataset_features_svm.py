import os
import sys
import numpy as np
import cv2 as cv
from image_processing.feature_extraction import feature_extraction
from utils.CVPR import AnnotationParser
from image_processing.divide_image import resize_image


def extract_features_svm(classes_data_files, feature_dir, model_dir, roi_width=64, roi_height=128, generate_labels=False, flip=False, border=0, debug=False, debug_label=''):
    classes_labels = {}
    label_counter = 0
    if flip:
        print "FLIP: true"

    used_classes_data_files = dict()
    unused_classes_data_files = dict()
    per_class_used_annotations = dict()
    per_class_unused_annotations = dict()
    for cl in classes_data_files.keys():
        used_classes_data_files[cl] = dict()
        unused_classes_data_files[cl] = dict()
        per_class_used_annotations[cl] = 0
        per_class_unused_annotations[cl] = 0
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
                    image = annotation_parser.get_image_from_roi(roi, border=border)
                    image = resize_image(image, model_width=roi_width, model_height=roi_height)
                    features_arrays.append(feature_extraction(image))
                    if flip:
                        image_flip = cv.flip(image, 1)
                        features_arrays.append(feature_extraction(image_flip))
                    if fi not in used_classes_data_files[cl].keys():
                        used_classes_data_files[cl][fi] = 0
                    used_classes_data_files[cl][fi] += 1
                    per_class_used_annotations += 1
                except Exception:
                    if fi not in unused_classes_data_files[cl].keys():
                        unused_classes_data_files[cl][fi] = 0
                    unused_classes_data_files[cl][fi] += 1
                    per_class_used_annotations[cl] += 1

            save_features_to_npz(features_arrays, features_file_path)
        classes_labels[label_counter] = cl
        label_counter += 1
    if generate_labels:
        with open(os.path.join(model_dir, 'classes_labels.txt'), 'w+') as classes_labels_file:
            for key in sorted(classes_labels.keys()):
                classes_labels_file.write(str(classes_labels[key]) + "\t" + str(key) + "\n")
    if debug:
        print "Writing debug info"
        with open(os.path.join(os.path.join(model_dir, debug_label), 'per_class_used_annotations.txt'), 'w+') as pca:
            for cl in per_class_used_annotations.keys():
                pca.write(str(cl) + " : " + str(per_class_used_annotations[cl])+"\n")
        with open(os.path.join(os.path.join(model_dir, debug_label), 'per_class_unused_annotations.txt'), 'w+') as pca:
            for cl in per_class_unused_annotations.keys():
                pca.write(str(cl) + " : " + str(per_class_unused_annotations[cl]) + "\n")

        for cl in per_class_used_annotations.keys():
            with open(os.path.join(os.path.join(model_dir, debug_label), str(cl)+'_used_annotation.txt'), 'w+') as ua:
                for fi in used_classes_data_files[cl].keys():
                    ua.write(str(used_classes_data_files[cl][fi]) + "\t" + str(fi) + "\n")
            with open(os.path.join(os.path.join(model_dir, debug_label), str(cl) + '_unused_annotation.txt'), 'w+') as ua:
                for fi in used_classes_data_files[cl].keys():
                    ua.write(str(unused_classes_data_files[cl][fi]) + "\t" + str(fi) + "\n")
        print "End Writing debug info"


def save_features_to_npz(feat_array, feat_file_path):
    if os.path.exists(feat_file_path):
        raise IOError('Error: ' + feat_file_path + ' already exists!')

    np.savez(feat_file_path, feat_array)


def load_features_from_npz(npz_file_path):
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
    if model_dir[len(model_dir)-1:] != '/':
        model_dir += '/'
    if not os.path.exists(model_dir):
        raise IOError("ERROR: "+model_dir+" no such file or directory")

    classes_path = model_dir + 'classes.txt'
    classes_training_files_path = model_dir + 'training_data/'
    classes = []
    classes_training_files_lists_dict = dict()

    # Read selected classes
    with open(classes_path, 'r') as classes_file:
        for line in classes_file:
            c = line.replace('\n', '').strip()
            classes.append(c)
            classes_training_files_lists_dict[c] = dict()

    # Read annotation files locations for each class
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
        print '(2) --width=W ;roi width (default 64)'
        print '(3) --height=H ;roi height (default 128)'
        print '(4) --border=B ;border around the image (default 0)'
        print '(5) --debug produce debug output in debug_testing, debug_training folders'
        exit(1)

    model_dir = sys.argv[1]
    classes_dict = None
    classes_test_dict = None
    if model_dir[len(model_dir) - 1:] != '/':
        model_dir += '/'

    roi_width = 64
    roi_height = 128
    border = 0
    debug = False
    i = 0
    if len(sys.argv) > 2:
        print "****Selected Parameters****"
        for i in range(2, len(sys.argv)):
            param = str(sys.argv[i])
            if "width" in param:
                roi_width = int(param.split("=")[1])
                print "-roi_width="+str(roi_width)
            elif "height" in param:
                roi_height = int(param.split("=")[1])
                print "-roi_height" + str(roi_height)
            elif "border" in param:
                border = int(param.split("=")[1])
                print "-border=" + str(border)
            elif "debug" in param:
                debug = True
                print "-debug Active"
        print "****End Selected Parameters****"

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

    if debug:
        debug_training_path = os.path.join(model_dir, "debug_training")
        if not os.path.exists(debug_training_path):
            os.mkdir(debug_training_path)
        debug_testing_path = os.path.join(model_dir, "debug_testing")
        if not os.path.exists(debug_testing_path):
            os.mkdir(debug_testing_path)
        extract_features_svm(classes_dict, training_data_feat_dir, model_dir, roi_width, roi_height, True, flip=True, border=border, debug=True, debug_label="debug_training")
        extract_features_svm(classes_test_dict, testing_data_feat_dir, model_dir, roi_width, roi_height, border=border, debug=True, debug_label="debug_testing")
    else:
        extract_features_svm(classes_dict, training_data_feat_dir, model_dir, roi_width, roi_height, True, flip=True, border=border)
        extract_features_svm(classes_test_dict, testing_data_feat_dir, model_dir, roi_width, roi_height, border=border)