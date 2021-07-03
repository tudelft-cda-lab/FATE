from load_data import moons, iris, xor, breast_cancer, diabetes, ijcnn1, covtype, higgs, mnist_2_6, groot, \
    covtype_multi, mnist, fashion_mnist, webspam, vowel

"""
Simple
"""

_moons = {
    "full_name": "Moons",
    "difficult_for_milp": False,
    "data": moons,
    "num_classes": 2,
    "num_features": 2,
    "RF": {
        "num_est": 10,
        "max_depth": 10,
        "max_leaves": None,
        "target_acc": 0.99,
        "max_iter": 100,
    },
    "GB": {
        "num_est": 10,
        "max_depth": 6,
        "max_leaves": None,
        "target_acc": 0.99,
        "max_iter": 100,
    }
}

_iris = {
    "full_name": "Iris",
    "difficult_for_milp": False,
    "data": iris,
    "num_classes": 3,
    "num_features": 4,
    "RF": {
        "num_est": 4,
        "max_depth": 5,
        "max_leaves": 10,
        "target_acc": 0.99,
        "max_iter": 10,
    },
    "GB": {
        "num_est": 2,
        "max_depth": 2,
        "max_leaves": 4,
        "target_acc": 0.99,
        "max_iter": 10,
    }
}

_xor = {
    "full_name": "XOR",
    "difficult_for_milp": False,
    "data": xor,
    "num_classes": 2,
    "num_features": 2,
    "RF": {
        "num_est": 10,
        "max_depth": None,
        "max_leaves": None,
        "target_acc": 0.95,
        "max_iter": 100,
    },
    "GB": {
        "num_est": 10,
        "max_depth": None,
        "max_leaves": None,
        "target_acc": 0.95,
        "max_iter": 100,
    }
}

"""
Binary classification
"""
_breast_cancer = {
    "full_name": "Breast-cancer",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer",
    "description": "Winsconcin Breast-Cancer database.",
    "difficult_for_milp": False,
    "epsilon": 0.5,
    "mutation_chance": 0.5,
    "data": breast_cancer,
    "num_classes": 2,
    "num_features": 10,
    "RF": {
        "num_est": 4,
        "max_depth": 6,
        "max_leaves": None,
        "target_acc": 0.974,
        "max_iter": 100,
    },
    "GB": {
        "num_est": 4,
        "max_depth": 6,
        "max_leaves": None,
        "target_acc": 0.964,
        "max_iter": 100,
    }
}

_diabetes = {
    "full_name": "Diabetes",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes",
    "description": "AIM '94 Diabetes dataset.",
    "difficult_for_milp": False,
    "epsilon": 0.1,
    "mutation_chance": 0.05,
    "data": diabetes,
    "num_classes": 2,
    "num_features": 8,
    "RF": {
        "num_est": 25,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.775,
        "max_iter": 25,
    },
    "GB": {
        "num_est": 20,
        "max_depth": 5,
        "max_leaves": None,
        "target_acc": 0.773,
        "max_iter": 100,
    }
}

_ij = {
    "full_name": "IJCNN1",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1",
    "description": "IJCNN 2001 neural network competition dataset.",
    "difficult_for_milp": False,
    "epsilon": 0.05,
    "mutation_chance": 0.05,
    "data": ijcnn1,
    "num_classes": 2,
    "num_features": 22,
    "RF": {
        "num_est": 100,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.919,
        "max_iter": 25,
    },
    "GB": {
        "num_est": 60,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.98,
        "max_iter": 10,
    }
}

_cov = {
    # covertype binary classification
    "full_name": "Covertype Binary",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary",
    "description": "",
    "difficult_for_milp": True,
    "epsilon": 0.05,
    "mutation_chance": 0.05,
    "data": covtype,
    "num_classes": 2,
    "num_features": 54,
    "RF": {
        "num_est": 160,
        "max_depth": 10,
        "max_leaves": None,
        "target_acc": 0.8,  # multi-class should have > 0.745, binary should be easier?
        "max_iter": 3,
    },
    "GB": {
        "num_est": 80,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.877,
        "max_iter": 1,
    }
}

_higgs = {
    "full_name": "Higgs",
    "link": "https://www.openml.org/d/23512",
    "description": "The original, full HIGGS dataset is too big for our hardware to train a model. We thus used "
                   "this smaller version with 98050 instances.",
    "difficult_for_milp": True,
    "epsilon": 0.025,
    "mutation_chance": 0.05,
    "data": higgs,
    "num_classes": 2,
    "num_features": 28,
    "RF": {
        "num_est": 300,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.702,
        "max_iter": 3,
    },
    "GB": {
        "num_est": 300,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.76,
        "max_iter": 1,
    }
}

_26 = {
    "full_name": "MNIST 2 vs 6",
    "link": None,
    "description": "MNIST with only the '2' and '6' instances. Easier to classify, more difficult to attack than the "
                   "original MNIST set.",
    "difficult_for_milp": False,
    "data": mnist_2_6,
    "num_classes": 2,
    "num_features": 784,
    "mutation_chance": 0.5,
    "RF": {
        "num_est": 1000,
        "max_depth": 4,
        "max_leaves": None,
        "target_acc": 0.963,
        "max_iter": 2,
        "time": 10,
    },
    "GB": {
        "num_est": 1000,
        "max_depth": 4,
        "max_leaves": None,
        "target_acc": 0.998,
        "max_iter": 1,
        "time": 10,
    }
}

_webspam = {
    "full_name": "Webspam",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam",
    "description": "Unigram over web pages that are created to manipulate search engines and decieve web users by "
                   "De Wang, Danesh Irani, and Calton Pu. \"Evolutionary Study of Web Spam: Webb Spam Corpus 2011 "
                   "versus Webb Spam Corpus 2006\". In Proc. of 8th IEEE International Conference on Collaborative "
                   "Computing: Networking, Applications and Worksharing (CollaborateCom 2012). Pittsburgh, "
                   "Pennsylvania, United States, October 2012.",
    "difficult_for_milp": True,  # GB: 85 / RF 160 sec per victim, but 500 victims
    "data": webspam,
    "num_classes": 2,
    "num_features": 256,
    "mutation_chance": 0.05,
    "RF": {
        "num_est": 100,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.999,
        "max_iter": 1,
    },
    "GB": {
        "num_est": 100,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.999,
        "max_iter": 1,
    }
}

_groot = {
    "full_name": "GROOT - MNIST 2 vs 6",
    "link": None,
    "description": "Robustly trained GROOT classifier on the MNIST 2 vs 6 problem.",
    "difficult_for_milp": True,
    "data": groot,
    "num_classes": 2,
    "num_features": 784,
    "mutation_chance": 0.5,
    "RF": {
        "num_est": None,
        "max_depth": None,
        "max_leaves": None,
        "target_acc": None,
        "max_iter": None,
        "time": 10,
    }
}

"""
Multi-class
"""
_vowel = {
    "full_name": "Vowel",
    "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#vowel",
    "description": "",
    "difficult_for_milp": False,  # GB: 60 / RF 62 sec per victim, 208/265 victims
    "data": vowel,
    "num_classes": 11,
    "num_features": 10,
    "mutation_chance": 0.1,
    "RF": {
        "num_est": 50,
        "max_depth": 10,
        "max_leaves": None,
        "target_acc": 0.999,
        "max_iter": 1,
    },
    "GB": {
        "num_est": 50,
        "max_depth": 10,
        "max_leaves": None,
        "target_acc": 0.999,
        "max_iter": 1,
    }
}


_cov_m = {
    "full_name": "Covertype Multi",
    "link": "https://www.openml.org/d/150",
    "description": "Multi-class Covertype. Classes other than 0 and 1 are much less represented.",
    "difficult_for_milp": True,
    # covertype multi-class classification
    "data": covtype_multi,
    "num_classes": 7,
    "num_features": 54,
    # RF model is very big in size
    "RF": {
        "num_est": 160,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.745,
        "max_iter": 3,
    },
    "GB": {
        "num_est": 160,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.877,
        "max_iter": 1,
    }
}

_mnist = {
    "full_name": "MNIST",
    "link": "https://deepai.org/dataset/mnist",
    "description": "Image recognition / Digit classification challenge.",
    "difficult_for_milp": True,
    "data": mnist,
    "num_classes": 10,
    "num_features": 784,
    "mutation_chance": 0.5,
    "RF": {
        "num_est": 400,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.907,
        "max_iter": 10,
        "time": 10,
    },
    "GB": {
        "num_est": 400,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.98,
        "max_iter": 1,
        "time": 10,
    }
}

_fmnist = {
    "full_name": "FMNIST",
    "link": "https://github.com/zalandoresearch/fashion-mnist",
    "description": "More difficult drop-in replacement for MNIST created by Zalando Research.",
    "difficult_for_milp": True,
    "data": fashion_mnist,
    "num_classes": 10,
    "num_features": 784,
    "mutation_chance": 0.5,
    "RF": {
        "num_est": 400,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.823,
        "max_iter": 1,
        "time": 10,
    },
    "GB": {
        "num_est": 400,
        "max_depth": 8,
        "max_leaves": None,
        "target_acc": 0.903,
        "max_iter": 1,
        "time": 10,
    }
}


MODELS_BINARY = {
    "BC": _breast_cancer,
    "DIABETES": _diabetes,
    "IJ": _ij,
    "COV": _cov,
    "WEB": _webspam,
    "HIGGS": _higgs,
    "26": _26,
    # "GROOT": _groot,
}

MODELS_MULTI = {
    "VOWEL": _vowel,
    # "COV-M": _cov_m,
    "MNIST": _mnist,
    "FMNIST": _fmnist,
}

MODELS_SIMPLE = {
    # "XOR": _xor,
    # "MOONS": _moons,
    # "IRIS": _iris,
}

MODEL_SETTINGS = {**MODELS_BINARY, **MODELS_MULTI, **MODELS_SIMPLE}
DATASETS = list(MODEL_SETTINGS.keys())


def get_full_name(dataset_name):
    return MODEL_SETTINGS[dataset_name]["full_name"]


def get_num_classes(dataset_name):
    return MODEL_SETTINGS[dataset_name]["num_classes"]


def get_num_features(dataset_name):
    return MODEL_SETTINGS[dataset_name]["num_features"]


def get_mutation_chance(dataset_name):
    from constants import DEFAULT_MUTATE_CHANCE, FORCE_DEFAULT_MUTATION_CHANCE
    if FORCE_DEFAULT_MUTATION_CHANCE:
        return DEFAULT_MUTATE_CHANCE
    return MODEL_SETTINGS[dataset_name].get("mutation_chance", DEFAULT_MUTATE_CHANCE)


def get_epsilon(dataset_name):
    from constants import DEFAULT_EPSILON, FORCE_DEFAULT_EPSILON
    if FORCE_DEFAULT_EPSILON:
        return DEFAULT_EPSILON
    return MODEL_SETTINGS[dataset_name].get("epsilon", DEFAULT_EPSILON)


def get_entropic(dataset_name):
    return str(int(dataset_name not in ['MNIST', 'FMNIST']))


def get_dataset_description(dataset_name):
    return MODEL_SETTINGS[dataset_name].get('description', '')


def get_link(dataset_name):
    return MODEL_SETTINGS[dataset_name].get('link', None)


def get_execution_time(dataset_name, mt):
    from constants import DEFAULT_TIME_PER_POINT, LIMIT_TIME
    m = 1
    if not LIMIT_TIME:
        m = 10
    return MODEL_SETTINGS[dataset_name][mt].get('time', DEFAULT_TIME_PER_POINT) * m


def is_difficult_for_milp(dataset_name):
    return MODEL_SETTINGS[dataset_name]["difficult_for_milp"]
