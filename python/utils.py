import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from external.util import sklearn_forest_to_xgboost_json, sklearn_booster_to_xgboost_json


def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()


def write_file(filename, contents):
    with open(filename, 'w+') as file:
        file.write(contents)


def save(m, filename):
    pickle.dump(m, open(filename, 'wb+'))


def load(filename):
    return pickle.load(open(filename, 'rb'))


def makedir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def make_or_empty_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    makedir(dir_path)


def parent(dir_path):
    return str(Path(dir_path).parent)


def remove_file(filename):
    remove_files([filename])


def remove_files(filenames):
    for filename in filenames:
        try:
            os.remove(filename)
        except FileNotFoundError:
            continue


def log_likelihood(val: float):
    return np.exp(val)/(1+np.exp(val))


def softmax(vals):
    s = sum([np.exp(v) for v in vals])
    return [np.exp(v)/s for v in vals]


def to_json(clf, filename, mt):
    if mt == 'RF':
        sklearn_forest_to_xgboost_json(clf, filename)
    else:
        sklearn_booster_to_xgboost_json(clf, filename)
