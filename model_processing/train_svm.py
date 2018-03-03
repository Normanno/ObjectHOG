from sklearn import svm
from sklearn.externals import joblib
from extract_dataset_features_svm import load_features_from_npz
import sys
import os


def train_svm_CVPR(classes, model_dir, feats_dir):
    """
    >train_svm(classes, model_dir, feats_dir)
    :param classes: dictionary label:class_name
    :param model_dir:
    :param feats_dir:
    :return:
    """
    X_data = []
    X_labels = []
    class_name = ''
    class_label = ''
    for cl in classes.keys():
        class_name = classes[cl]
        class_label = cl
        class_feats_dir = os.path.join(feats_dir, class_name)
        print ' class ' + class_name + ' - ' + str(class_label)
        for f in os.listdir(feats_dir + "/" + class_name):
            if f.endswith(".npz"):
                feats = load_features_from_npz(os.path.join(class_feats_dir, f))
                feats_number = len(feats)
                reshaped_feats = list()
                for i in range(0, feats_number):
                    r_feats = feats[i].reshape(1, -1)
                    reshaped_feats.extend(r_feats)

                X_data.extend(reshaped_feats)
                X_labels.extend([class_label] * feats_number)
    print '---[Starting training]---'
    print 'kernel : rbf'
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_data, X_labels)
    print '---[End of training]---'
    out_model = os.path.join(model_dir, "model.pkl")
    joblib.dump(clf, out_model)

if __name__ == "__main__":
    print "train"
    if len(sys.argv) < 2:
        print '***** Error *****'
        print 'This program requires 1 arguments ('+str(len(sys.argv)-1)+') given :'
        print '- model src directory, (root of the model config dir)'
        exit(1)

    model_dir = sys.argv[1]
    if model_dir[len(model_dir) - 1:] != '/':
        model_dir += '/'

    classes_labels = dict()
    with open(os.path.join(model_dir, 'classes_labels.txt'), 'r') as lf:
        for line in lf:
            fields = str(line).replace('\n', '').split('\t')
            classes_labels[int(fields[1])] = fields[0]

    train_svm_CVPR(classes_labels, model_dir, os.path.join(model_dir, 'training_data_feats_w64_h128'))
