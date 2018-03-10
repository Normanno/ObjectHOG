from sklearn import svm
from sklearn.externals import joblib
from extract_dataset_features_svm import load_features_from_npz
import numpy as np
import sys
import os


def test_model(classes, model_dir, feats_dir):
    print 'testing'
    model_path = os.path.join(model_dir, 'model.pkl')
    clf = joblib.load(model_path)
    X_data = []
    X_labels = []
    classes_stats = dict()
    class_name = ''
    class_label = ''
    for cl in classes.keys():
        classes_stats[cl] = {"correct": 0, "error": 0, "classification: ": dict()}
        classes_stats[cl]["classification"] = dict()
        for cli in classes.keys():
            classes_stats[cl]["classification"][cli] = 0
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
    print '---[Start testing]---'
    Y_labels = clf.predict(X_data)
    predict_probabilities = clf.predict_proba(X_data)
    print "predict probabilities : " + str(predict_probabilities)
    if len(Y_labels) != len(X_labels):
        print 'Error : \n-Y_labels: ' + str(len(Y_labels)) + "\n-X_labels:" + str(len(X_labels))
    else:
        print "---[Computing stats]---"
        for i in range(0, len(X_labels)):
            rl = X_labels[i]
            pl = Y_labels[i]
            if rl == pl:
                classes_stats[rl]["correct"] += 1
                classes_stats[rl]["classification"][rl] += 1
            else:
                classes_stats[rl]["error"] += 1
                classes_stats[rl]["classification"][pl] += 1
        print "---[Saving stats]---"
        stats_file = os.path.join(model_dir, 'stats.txt')
        with open(stats_file, 'w+') as sf:
            for cl in classes.keys():
                sf.write("-"+classes[cl]+"\n")
                sf.write("--correct:\t" + str(classes_stats[cl]["correct"]) + "\n")
                sf.write("--error:\t" + str(classes_stats[cl]["error"]) + "\n")
                sf.write("--classified as:\n")
                for cli in classes.keys():
                    sf.write("---"+classes[cli]+":\t"+str(classes_stats[cl]["classification"][cli])+"\n")
    print '---[End of testing]---'





if __name__ == "__main__":
    print "train"
    if len(sys.argv) < 2:
        print '***** Error *****'
        print 'This program requires 1 arguments ('+str(len(sys.argv)-1)+') given :'
        print '- model src directory, (root of the model config dir)'
        exit(1)

    model_dir = sys.argv[1]
    classes_dict = None
    if model_dir[len(model_dir) - 1:] != '/':
        model_dir += '/'

    classes_labels = dict()
    with open(os.path.join(model_dir, 'classes_labels.txt'), 'r') as lf:
        for line in lf:
            fields = str(line).replace('\n', '').split('\t')
            classes_labels[int(fields[1])] = fields[0]

    test_model(classes_labels, model_dir, os.path.join(model_dir, 'testing_data_feats_w64_h128'))
