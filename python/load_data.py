import random
from os.path import join

import numpy as np
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import idx2numpy

from constants import DATA_DIR, TEST_FRACTION, LIB_SVM_DIR, OPEN_ML_DIR, NUM_SAMPLES, CONSISTENT_DRAWS
from utils import read_file, load


"""
Synthetic datasets
"""


def xor():
    x = []
    y = []
    for _ in range(NUM_SAMPLES):
        x1 = random.random()
        x2 = random.random()
        out = (round(x1) + round(x2)) % 2
        x.append([x1, x2])
        y.append(int(out))

    x = _scale_features(x)
    return _split(x, y)


def visualise_moons(x, y):
    df = DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()


def moons():
    x, y = make_moons(n_samples=NUM_SAMPLES, noise=0.3, random_state=0)
    y = [int(e) for e in y]
    # visualise_moons(x, y)
    x = _scale_features(x)
    return _split(x, y)


def iris():
    x, y = load_iris(return_X_y=True)
    x = _scale_features(x)
    return _split(x, y)


"""
LIB-SVM datasets
"""


def breast_cancer():
    # from sklearn.datasets import load_breast_cancer
    # return load_breast_cancer(return_X_y=True)
    entries = _read_libsvm_file(join(LIB_SVM_DIR, "breast_cancer-libsvm"), 'BC')
    x, y = _parse_libsvm_entries(entries, encoding='2/4')
    return _split(x, y)


def diabetes():
    # from sklearn.datasets import load_diabetes
    # return load_diabetes(return_X_y=True)
    entries = _read_libsvm_file(join(LIB_SVM_DIR, "diabetes-libsvm"), 'DIABETES')
    x, y = _parse_libsvm_entries(entries, encoding='-1/1')
    return _split(x, y)


def ijcnn1():
    training_entries = _read_libsvm_file(join(LIB_SVM_DIR, "ijcnn1-libsvm-training"), 'IJ')
    testing_entries = _read_libsvm_file(join(LIB_SVM_DIR, "ijcnn1-libsvm-testing"), 'IJ')
    x_train, y_train = _parse_libsvm_entries(training_entries, encoding='-1/1')
    x_test, y_test = _parse_libsvm_entries(testing_entries, encoding='-1/1')
    return x_train, x_test, y_train, y_test


def covtype():
    filename = join(LIB_SVM_DIR, "covtype.libsvm.binary")
    entries = _read_libsvm_file(filename, 'COV')
    x, y = _parse_libsvm_entries(entries, encoding='1/2')
    return _split(x, y)


def webspam():
    filename = join(LIB_SVM_DIR, "webspam_wc_normalized_unigram.svm")
    entries = _read_libsvm_file(filename, 'WEB')
    x, y = _parse_libsvm_entries(entries, encoding='-1/1')
    return _split(x, y)


def vowel():
    filename_train = join(LIB_SVM_DIR, "vowel")
    entries_train = _read_libsvm_file(filename_train, 'VOWEL')
    x_train, y_train = _parse_libsvm_entries(entries_train, encoding='0+')
    filename_test = join(LIB_SVM_DIR, "vowel.test")
    entries_test = _read_libsvm_file(filename_test, 'VOWEL')
    x_test, y_test = _parse_libsvm_entries(entries_test, encoding='0+')
    return x_train, x_test, y_train, y_test


def _read_libsvm_file(filename: str, dataset_name: str):
    from datasets import get_num_features
    num_features = get_num_features(dataset_name)
    with open(filename, 'r') as file:
        content = file.read()
    lines = content.splitlines()
    parsed = []
    for line in lines:
        parsed_line = np.zeros(num_features+1)
        fields = line.split(" ")
        counter = 0
        for field in fields:
            if field == "":
                continue
            if ":" in field:
                splitted = field.split(":")
                parsed_line[int(splitted[0])] = float(splitted[1])
            else:
                if counter != 0:
                    raise ValueError('Counter is not 0 for label')
                # Class label
                parsed_line[0] = field
            counter += 1
        parsed.append(parsed_line)
    return parsed


def _parse_libsvm_entries(entries, encoding):
    """
    Parses libsvm encoded class labels for binary classification and separates them from the features.
    Returns scaled features.
    For encoding, choose on of:
    - '-1/1': -1/1 as labels
    - '1/2': 1/2 as labels
    - '2/4': 2/4 as labels
    - '0+': labels are consecutive integers (0 indexed)
    """
    x = []
    y = []
    for entry in entries:
        if encoding == '-1/1':
            class_label = int((float(entry[0])+1)/2)
        elif encoding == '1/2':
            class_label = int(entry[0])-1
        elif encoding == '2/4':
            class_label = int(float(entry[0])/2)-1
        elif encoding == '0+':
            class_label = int(entry[0])
        else:
            raise ValueError(f'Unknown encoding option {encoding}')

        features = entry[1:]
        x.append(features)
        y.append(class_label)
    x = _scale_features(x)
    return x, y


"""
OPEN-ML datasets
"""


def covtype_multi():
    # Source: https://www.openml.org/d/150
    filename = join(OPEN_ML_DIR, "covtype_openml.csv")
    from datasets import get_num_features
    num_features = get_num_features('COV-M')

    with open(filename, 'r') as file:
        lines = file.readlines()

    x = []
    y = []
    for line in lines[1:]:
        # First row are column labels
        splitted = line.split(',')
        if len(splitted) != num_features+1:
            raise ValueError(f"Line is {len(splitted)} long")
        x.append([float(e) for e in splitted[:-1]])
        y.append(int(splitted[-1]) - 1)  # Last column is the target column
    x = _scale_features(x)
    return _split(x, y)


def higgs():
    # Original dataset is too big
    # Source: https://www.openml.org/d/23512
    entries = read_file(join(OPEN_ML_DIR, "higgs_2_openml.csv")).splitlines()
    x = []
    y = []
    for line in entries[1:]:
        # First line has the column labels
        entry = line.split(",")
        class_label = int(entry[0])
        features = []
        for feature in entry[1:]:
            try:
                features.append(float(feature))
            except ValueError:
                # Treat missing value as 0
                features.append(float(0))
        x.append(features)
        y.append(class_label)
    x = _scale_features(x)
    return _split(x, y)


"""
Image data
"""


def mnist():
    train_filename = join(DATA_DIR, "mnist/train-images-idx3-ubyte")
    test_filename = join(DATA_DIR, "mnist/t10k-images-idx3-ubyte")
    train_labels_filename = join(DATA_DIR, "mnist/train-labels-idx1-ubyte")
    test_labels_filename = join(DATA_DIR, "mnist/t10k-labels-idx1-ubyte")
    return _load_mnist_type(train_filename, train_labels_filename, test_filename, test_labels_filename)


def mnist_2_6():
    x_train, x_test, y_train, y_test = mnist()
    x_train_filtered = []
    y_train_filtered = []
    for fs, label in zip(x_train, y_train):
        if label in [2, 6]:
            x_train_filtered.append(fs)
            y_train_filtered.append(0 if label == 2 else 1)
    x_test_filtered = []
    y_test_filtered = []
    for fs, label in zip(x_test, y_test):
        if label in [2, 6]:
            x_test_filtered.append(fs)
            y_test_filtered.append(0 if label == 2 else 1)
    return x_train_filtered, x_test_filtered, y_train_filtered, y_test_filtered


def fashion_mnist():
    train_filename = join(DATA_DIR, "fmnist/train-images-idx3-ubyte")
    test_filename = join(DATA_DIR, "fmnist/t10k-images-idx3-ubyte")
    train_labels_filename = join(DATA_DIR, "fmnist/train-labels-idx1-ubyte")
    test_labels_filename = join(DATA_DIR, "fmnist/t10k-labels-idx1-ubyte")
    return _load_mnist_type(train_filename, train_labels_filename, test_filename, test_labels_filename)


def groot():
    x_train = load(join(DATA_DIR, 'mnist_2_6/mnist_2_6_x_train.sav'))
    y_train = load(join(DATA_DIR, 'mnist_2_6/mnist_2_6_y_train.sav'))
    x_test = load(join(DATA_DIR, 'mnist_2_6/mnist_2_6_x_test.sav'))
    y_test = load(join(DATA_DIR, 'mnist_2_6/mnist_2_6_y_test.sav'))
    return x_train, x_test, y_train, y_test


def _load_mnist_type(train_filename, train_labels_filename, test_filename, test_labels_filename):
    train_data = idx2numpy.convert_from_file(train_filename)
    test_data = idx2numpy.convert_from_file(test_filename)
    train_labels = idx2numpy.convert_from_file(train_labels_filename)
    test_labels = idx2numpy.convert_from_file(test_labels_filename)
    x_train = _scale_features(train_data, per_feature=False)
    y_train = [int(e) for e in train_labels]
    x_test = _scale_features(test_data, per_feature=False)
    y_test = [int(e) for e in test_labels]
    return x_train, x_test, y_train, y_test


"""
Generic util functions
"""


def _split(x, y):
    if CONSISTENT_DRAWS:
        return train_test_split(x, y, test_size=TEST_FRACTION, random_state=0)

    return train_test_split(x, y, test_size=TEST_FRACTION)


def _scale_features(features, per_feature=True):
    """
    Scales per feature by default.
    If per_feature is False, is scales over all features at the same time.
    """
    scaler = MinMaxScaler()
    if per_feature:
        return scaler.fit_transform(features)  # Scale per feature

    num_entries = len(features)
    features = np.asarray(features).reshape(-1, 1)  # Reshape to single feature (trick for the scaler)
    transformed = scaler.fit_transform(features)
    return transformed.reshape(num_entries, -1)  # Reshape to original dimensions


if __name__ == '__main__':
    import time
    from datasets import get_num_features
    from run_fate import generate_model_path
    _dn = 'MNIST'
    _c = load(generate_model_path(_dn, 'GB'))
    _nf = get_num_features(_dn)
    _xt = [[random.random() for _ in range(_nf)] for _ in range(100000)]
    _t_0 = time.time()
    _c.predict(_xt)
    print('Execution took ', time.time() - _t_0)
