import json
import os
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from jinja2 import FileSystemLoader, Environment
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import LIBFUZZER_TEMPLATE_PATH, BYTES_PER_FEATURE, ADV_DIR, USE_GAUSSIAN, \
    FEATURE_IMPORTANCE_BASED_ON_OCCURRENCE, USE_FEATURE_IMPORTANCE, FUZZ_ONE_POINT_PER_INSTANCE, \
    USE_THRESHOLDS_FOR_MUTATION, SMALL_PERTURBATION_THRESHOLD, USE_CROSSOVER, NUM_THREADS, ALSO_MUTATE_BIGGEST, \
    THRESHOLD_DIGITS, MUTATE_BIGGEST_CHANCE, BIAS_MUTATE_BIG_DIFFS, MUTATE_LESS_WHEN_CLOSER, \
    AE_CHECK_IN_MUTATE, AE_MUTATE_TOWARDS_VICTIM, STEEP_CURVE, USE_CUSTOM_MUTATOR, FUZZER, AFLPP_TEMPLATE_PATH, \
    AFL_MUTATE_FILENAME, IS_AE_CHANCE, NUM_CYCLES_IN_LOOP, AFLGO_TEMPLATE_PATH, MUTATE_TEMPLATE_PATH, IS_AE_FAKE, \
    TEST_OUTSIDE_FUZZER, USE_WAS_AE, MINIMIZE_THRESHOLD_LIST, MUTATE_DEPTH, CROSSOVER_CHANCE, NUM_RUNS, POPULATION_SIZE, \
    CROSSOVER_RANDOM_CHANCE, BLACKBOX
from datasets import get_mutation_chance, get_execution_time
from generate_code import _generate_check_function_calls, _generate_estimator_calls, _generate_adversarial_checks, \
    _array_initialize, _generate_threshold_init, _generate_threshold_vectors, generate_predict_probs
from tree import TreeNode, MultiTreeNode
from utils import log_likelihood, softmax


class BaseForest(ABC):
    def __init__(self, trees: Union[List['TreeNode'], List['MultiTreeNode']], num_classes, feature_importances=None):
        self.trees = trees
        self.num_classes = num_classes
        self.feature_importances = feature_importances

    @property
    def num_estimators(self):
        return len(self.trees)

    def predict(self, X):
        probas = self.predict_proba(X)
        return [np.argmax(p) for p in probas]

    def predict_proba(self, X):
        return [self._predict_single_proba(fs) for fs in X]

    @abstractmethod
    def _predict_single_proba(self, fs):
        pass

    @abstractmethod
    def generate_cc(self, *args, **kwargs):
        pass

    def walk(self):
        all_nodes = []
        for tree in self.trees:
            all_nodes += tree.walk()
        return all_nodes

    def feature_occurrences(self, num_features):
        all_nodes = self.walk()
        feature_indices = [n.feature_num for n in all_nodes]
        fi = np.zeros(num_features)
        for i in feature_indices:
            fi[i] += 1
        return fi

    def prior_mutate_chances(self, num_features, mutate_chance):
        fi = self.feature_occurrences(num_features)
        if not BLACKBOX:
            return [0 if e == 0 else mutate_chance for e in fi]
        else:
            return [mutate_chance for _ in fi]

    def choose_feature_importances(self, num_features):
        if USE_FEATURE_IMPORTANCE:
            if FEATURE_IMPORTANCE_BASED_ON_OCCURRENCE:
                fi = self.feature_occurrences(num_features)
            else:
                if self.feature_importances is None:
                    raise NotImplementedError("Feature importances can only be used in combination with sklearn models."
                                              " Please set FEATURE_IMPORTANCE_BASED_ON_OCCURRENCE to True or"
                                              " USE_FEATURE_IMPORTANCE to False.")
                fi = self.feature_importances
        else:
            fi = [1 for _ in range(num_features)]

        if len(fi) != num_features:
            raise ValueError("Number of feature importances does not match number of features")

        sum_fi = sum(fi)
        fi = np.array(fi, dtype='float')
        fi /= sum_fi  # From raw counts / probabilities to chances

        return fi

    def thresholds_per_feature(self, num_features):
        # from result import stats
        threshold_set_per_feature = [set() for _ in range(num_features)]  # set to remove duplicate thresholds
        all_nodes = self.walk()
        for node in all_nodes:
            # Rounding such that Thresholds which differ in less than THRESHOLD_DIGITS are not duplicated
            threshold_set_per_feature[node.feature_num].add(round(node.threshold, THRESHOLD_DIGITS))
        sorted_thres = [sorted(s) for s in threshold_set_per_feature]
        # print('before: ', sorted_thres[5])
        # print('before')
        # pprint(sorted_thres)
        # thres_len = [len(e) for e in sorted_thres]
        # print(stats(thres_len, name='Thresholds', stdout=True))

        if MINIMIZE_THRESHOLD_LIST:
            for l in sorted_thres:
                indices_to_remove = []
                val = - 1
                for i, e in enumerate(l):
                    if e - val < 0.0001:
                        indices_to_remove.append(i)
                    else:
                        val = e
                for ind in reversed(indices_to_remove):
                    del l[ind]

        # print('after')
        # thres_len = [len(e) for e in sorted_thres]
        # print(stats(thres_len, name='Thresholds', stdout=True))
        return sorted_thres

    def tree_functions_cc(self, indent_branches=False):
        return "\n".join([tree.tree_function_cc(i, self.num_classes, indent_branches)
                          for i, tree in enumerate(self.trees)])

    def _generate_cc_from_forest(self, filename, x_test, y_test, epsilon, num_mutations, forest_type, dataset_name,
                                 extra: dict):
        """
        x_test and y_test are not necessary when fuzzing 1 point at a time.
        They are artifacts from fuzzing with n objective AE functions
        """
        num_features = len(x_test[0])
        tree_functions = self.tree_functions_cc()
        mutate_chance = get_mutation_chance(dataset_name)

        file_loader = FileSystemLoader(os.path.dirname(__file__))
        env = Environment(loader=file_loader)
        context = {
            'tree_functions': tree_functions,
            'check_functions': _generate_adversarial_checks(x_test, y_test, self.num_classes, epsilon),
            'tree_function_calls': _generate_estimator_calls(self.num_estimators),
            'check_function_calls': _generate_check_function_calls(len(y_test)),
            'epsilon': epsilon,
            'expected_size': num_features * BYTES_PER_FEATURE,
            'num_features': num_features,
            'num_classes': self.num_classes,
            'predict_probs': generate_predict_probs(self.num_estimators),
            'adv_path': ADV_DIR,
            'num_mutations': num_mutations,
            'forest_type': forest_type,
            'mutate': 'GAUSSIAN' if USE_GAUSSIAN else 'RANDOM',
            'fuzz': 'ONE' if FUZZ_ONE_POINT_PER_INSTANCE else 'MULTIPLE',
            'nc': 1 if self.num_classes == 2 else self.num_classes,
            'mutate_chances': _array_initialize(self.prior_mutate_chances(num_features, mutate_chance)),
            'feature_importances': _array_initialize(self.choose_feature_importances(num_features)),
            'threshold_vectors': _generate_threshold_vectors(self.thresholds_per_feature(num_features)),
            'threshold_init': _generate_threshold_init(num_features),
            'use_thresholds': USE_THRESHOLDS_FOR_MUTATION,
            'use_crossover': USE_CROSSOVER,
            'small_perturbation': str(SMALL_PERTURBATION_THRESHOLD),
            'also_mutate_biggest': ALSO_MUTATE_BIGGEST,
            'mutate_biggest_chance': MUTATE_BIGGEST_CHANCE,
            'mutate_chance': mutate_chance,
            'mutate_less_when_closer': MUTATE_LESS_WHEN_CLOSER,
            'bias_big_diff_features': BIAS_MUTATE_BIG_DIFFS,
            'ae_mutate_towards_original': AE_MUTATE_TOWARDS_VICTIM,
            'use_feature_importances': USE_FEATURE_IMPORTANCE,
            'ae_check_in_mutate': AE_CHECK_IN_MUTATE,
            'steep_curve': STEEP_CURVE,
            'use_custom_mutator': USE_CUSTOM_MUTATOR,
            'is_ae_chance': IS_AE_CHANCE,
            'num_cycles_in_loop': NUM_CYCLES_IN_LOOP,
            'fake_is_ae': IS_AE_FAKE,
            'test_outside_fuzzer': TEST_OUTSIDE_FUZZER,
            'use_was_ae': USE_WAS_AE,
            'fuzzer': 'AFL' if FUZZER in ['AFL++', 'AFLGo'] else 'libFuzzer',  # libFuzzer also for honggfuzz
            'mutation_depth': MUTATE_DEPTH,
            'crossover_chance': CROSSOVER_CHANCE,
            'crossover_random_chance': CROSSOVER_RANDOM_CHANCE,
            'time_seconds': get_execution_time(dataset_name, forest_type)+1,
            'num_runs': NUM_RUNS,
            'population_size': POPULATION_SIZE,
        }
        context.update(extra)

        if FUZZER in ['libFuzzer', 'honggfuzz']:
            template = env.get_template(LIBFUZZER_TEMPLATE_PATH)
            output = template.render(context)
            with open(filename, 'w+') as file:
                file.write(output)
        else:
            if FUZZER in ['AFL++', 'AFLGo']:
                tpath = AFLPP_TEMPLATE_PATH if FUZZER == 'AFL++' else AFLGO_TEMPLATE_PATH
                target_template = env.get_template(tpath)
                output = target_template.render(context)
                with open(filename, 'w+') as file:
                    file.write(output)

                mutate_template = env.get_template(MUTATE_TEMPLATE_PATH)
                output = mutate_template.render(context)
                with open(AFL_MUTATE_FILENAME, 'w+') as file:
                    file.write(output)
            else:
                ...

    @staticmethod
    def train(t, x_train, x_test, y_train, y_test, num_estimators, max_depth, max_leaves, min_acc,
              max_iter, learning_rate=None):
        n = 0
        best_accuracy = 0
        best_clf = None
        while best_accuracy < min_acc and n < max_iter:
            n += 1
            print(f"Iteration {n}")
            if t == 'RF':
                clf = RandomForestClassifier(verbose=1, n_jobs=NUM_THREADS, n_estimators=num_estimators,
                                             max_depth=max_depth, max_leaf_nodes=max_leaves, min_samples_leaf=5,
                                             min_samples_split=10)
            else:
                clf = GradientBoostingClassifier(verbose=1, learning_rate=learning_rate, n_estimators=num_estimators,
                                                 max_depth=max_depth, max_leaf_nodes=max_leaves, min_samples_leaf=5,
                                                 min_samples_split=10)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_clf = clf
            print(f"Best acc: {round(best_accuracy, 4)}")

        print("Number of iterations: ", n)
        print("Accuracy: ", best_accuracy)
        return best_clf, best_accuracy


class RandomForest(BaseForest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict_single_proba(self, fs):
        sums = np.zeros(self.num_classes)
        for tree in self.trees:
            sums += tree.get_leaf_value(fs)
        sums /= self.num_estimators
        return sums

    def generate_cc(self, *args, **kwargs):
        self._generate_cc_from_forest(*args, **kwargs, forest_type='RF', extra={})

    @staticmethod
    def from_sklearn(clf, num_classes):
        return RandomForest([TreeNode.from_sklearn_classifier(e) for e in clf.estimators_], num_classes,
                            clf.feature_importances_)

    @staticmethod
    def from_file(filename, num_classes) -> 'RandomForest':
        with open(filename, 'r') as file:
            contents = file.read()
        return RandomForest.from_json(json.loads(contents), num_classes)

    @staticmethod
    def from_json(json_object, num_classes) -> 'RandomForest':
        trees = []
        if isinstance(json_object, list):
            for tree in json_object:
                trees.append(TreeNode.from_json(tree, 0))
        else:
            trees.append(TreeNode.from_json(json_object, 0))
        return RandomForest(trees, num_classes)


class GradientBoostingForest(BaseForest):
    def __init__(self, learning_rate: float, init_vals: List[float], *args, **kwargs):
        self.learning_rate = learning_rate
        self.init_vals = init_vals
        super().__init__(*args, **kwargs)

    def _predict_single_decision(self, fs):
        if self.num_classes == 2:
            total = self.init_vals[0]
            for tree in self.trees:
                total += self.learning_rate * tree.get_leaf_value(fs)[0]
        else:
            total = np.array(self.init_vals)
            for multi_tree in self.trees:
                total += multi_tree.get_leaf_value(fs) * self.learning_rate
        return total

    def _predict_single_proba(self, fs):
        decision = self._predict_single_decision(fs)
        if self.num_classes == 2:
            lv = log_likelihood(decision)
            return [1 - lv, lv]
        else:
            return softmax(decision)

    def generate_cc(self, *args, **kwargs):
        extra_context = {
            'learning_rate': self.learning_rate,
            'initial_prediction': self._to_init_pred_str(self.init_vals),
        }
        self._generate_cc_from_forest(*args, **kwargs, forest_type='GB', extra=extra_context)

    @staticmethod
    def _to_init_pred_str(vals):
        if len(vals) == 1:
            return "{"+",".join([str(e) for e in vals])+"}"
        else:
            base = ""
            for i, e in enumerate(vals):
                base += f"decisions[{i}] = {e};\n"
            return base

    @staticmethod
    def from_sklearn(clf, num_classes, learning_rate, init_vals) -> 'GradientBoostingForest':
        tree_nodes = []
        for e in clf.estimators_:
            # An estimator of Gradient Boosting Classifier is an ndarray with shape
            # (DecisionTreeRegressor, loss.K) loss.K is 1 for binary classification, otherwise num_classes
            if num_classes == 2:
                if len(e) != 1:
                    raise ValueError(f'There is more than one estimator for binary classification: {e}')
                tree_nodes.append(TreeNode.from_sklearn_regressor(e[0]))
            else:
                tree_nodes.append(MultiTreeNode([TreeNode.from_sklearn_regressor(r) for r in e]))

        return GradientBoostingForest(learning_rate, init_vals, tree_nodes, num_classes, clf.feature_importances_)
