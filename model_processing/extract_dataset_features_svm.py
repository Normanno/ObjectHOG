import sys
import os
import numpy as np
from image_processing.feature_extraction import feature_extraction



def extract_training_features_SVM(classes, classes_training_files, feature_dir):

    for cl in classes:
        for fi in classes_training_files[cl]:

            features = feature_extraction()




if __name__ == '__main__':
    print' dataset features extraction '
