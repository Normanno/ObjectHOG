import sys
import os


def select(src_dir, conf_dir, classes_file):
    print ''


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
