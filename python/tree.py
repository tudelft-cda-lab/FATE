from io import StringIO

import pydotplus
from os.path import join
from typing import List, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, DecisionTreeRegressor, export_graphviz

from constants import MODEL_TYPES, IMAGE_DIR
from datasets import get_num_classes, get_num_features
from utils import load


class MultiTreeNode:
    def __init__(self, tree_nodes: List['TreeNode']):
        self.tree_nodes = tree_nodes

    @property
    def num_classes(self):
        return len(self.tree_nodes)

    def walk(self):
        res = []
        for tree_node in self.tree_nodes:
            res += tree_node.walk()
        return res

    def get_leaf_value(self, fs):
        return np.array([t.get_leaf_value(fs)[0] for t in self.tree_nodes])

    def _call_function(self, n):
        indent = ""
        indent_line = " "
        function_base = f"double* tree_{n}(double fs[]) {{\n"
        nc = 1 if self.num_classes == 2 else self.num_classes
        function_base += f"{indent}static double res[{nc}];\n"
        for i in range(len(self.tree_nodes)):
            function_base += f'{indent}res[{i}] = tree_{n}_{i}(fs)[0];\n'
        return function_base + "return res;" + indent_line + "}"

    def tree_function_cc(self, n, _, indent_branches):
        trees = "\n".join([t.tree_function_cc(i, 1, indent_branches=indent_branches, n=n) for i, t in
                           enumerate(self.tree_nodes)])
        return trees + "\n" + self._call_function(n) + "\n"


class TreeNode:
    def __init__(self, node_id, depth, feature_num, threshold, yes_id, no_id, missing, is_leaf: bool,
                 child_left: Union['TreeNode', None], child_right: Union['TreeNode', None]):
        self.node_id = node_id
        self.depth = depth
        self.feature_num = feature_num
        self.threshold = threshold
        self.yes_id = yes_id
        self.no_id = no_id
        self.missing = missing
        self.is_leaf = is_leaf
        self.child_left = child_left
        self.child_right = child_right
        self._verify_parsing()

    def _verify_parsing(self):
        if not self.is_leaf:
            if self.child_left.node_id != self.yes_id:
                raise ValueError(f'Node {self.node_id} was not parsed correctly')
            if self.child_right.node_id != self.no_id:
                raise ValueError(f'Node {self.node_id} was not parsed correctly')

    def walk(self):
        res = [self]
        if not self.child_left.is_leaf:
            res += self.child_left.walk()
        if not self.child_right.is_leaf:
            res += self.child_right.walk()
        return res

    @property
    def indent(self):
        return "\t" * (self.depth + 1)

    def get_leaf_value(self, fs):
        if fs[self.feature_num] <= self.threshold:
            return self.child_left.get_leaf_value(fs)
        return self.child_right.get_leaf_value(fs)

    def _branch(self, indent_branches: bool):
        indent = self.indent if indent_branches else ""
        indent_line = "\n" if indent_branches else " "

        threshold = round(self.threshold, 7)  # This may cause float mis-classification, a known problem, no solution

        if_statement = f"{indent}if (fs[{self.feature_num}] <= {threshold}) {{"
        if_content = self.child_left._branch(indent_branches)
        if_branch = f"{if_statement}\n{if_content}\n{indent}}}"

        else_statement = f" else {{"
        else_content = self.child_right._branch(indent_branches)
        else_branch = f"{else_statement}\n{else_content}{indent_line}{indent}}}"

        return f"{if_branch}{else_branch}"

    def tree_function_cc(self, i: int, num_classes: int, indent_branches: bool = False, n: int = None) -> str:
        indent = "\t" if indent_branches else ""
        indent_line = "\n" if indent_branches else " "
        f_name = f'tree_{i}' if n is None else f'tree_{n}_{i}'
        function_base = f"double* {f_name}(double fs[]) {{\n"
        nc = num_classes  # on the safe side, for binary classification with GB only one value is returned
        function_base += f"{indent}static double res[{nc}];\n"
        return function_base + self._branch(indent_branches) + indent_line + "}"

    @staticmethod
    def from_sklearn_regressor(regressor: DecisionTreeRegressor) -> 'TreeNode':
        return TreeNode._from_sklearn_tree(regressor.tree_, 0, 0, 'GB')

    @staticmethod
    def from_sklearn_classifier(classifier: DecisionTreeClassifier) -> 'TreeNode':
        return TreeNode._from_sklearn_tree(classifier.tree_, 0, 0, 'RF')

    @staticmethod
    def _from_sklearn_tree(tree_, node_id, depth, t) -> 'TreeNode':
        is_leaf = tree_.feature[node_id] == _tree.TREE_UNDEFINED
        if is_leaf:
            if t not in MODEL_TYPES:
                raise ValueError(f'type {t} not recognised')
            if t == 'RF':
                class_amounts = tree_.value[node_id][0]
                sum_amounts = sum(class_amounts)
                probabilities = np.array([e / sum_amounts for e in class_amounts])
                return Leaf(node_id, probabilities, depth)
            else:
                # raw_predictions = ndarray of shape (n_samples, n_classes) or (n_samples,)
                #             The decision function of the input samples, which corresponds to
                #             the raw values predicted from the trees of the ensemble . The
                #             order of the classes corresponds to that in the attribute
                #             :term:`classes_`. Regression and binary classification produce an
                #             array of shape [n_samples].
                # DecisionTreeRegressor
                # "prior": always predicts the class that maximizes the class prior
                #           (like "most_frequent") and ``predict_proba`` returns the class prior.
                # loss = deviance, learning_rate = 0.1

                class_amounts = tree_.value[node_id][0]
                if len(class_amounts) == 1:
                    probabilities = np.array([class_amounts[0]])
                else:
                    raise NotImplementedError  # Should not be possible
                return Leaf(node_id, probabilities, depth)

        threshold = tree_.threshold[node_id]
        feature_num = tree_.feature[node_id]
        yes_id = tree_.children_left[node_id]
        no_id = tree_.children_right[node_id]

        child_left = TreeNode._from_sklearn_tree(tree_, yes_id, depth + 1, t)
        child_right = TreeNode._from_sklearn_tree(tree_, no_id, depth + 1, t)

        return TreeNode(node_id, depth, feature_num, threshold, yes_id, no_id, None, is_leaf, child_left, child_right)

    @staticmethod
    def from_json(json_node: dict, depth) -> 'TreeNode':
        if "leaf" in json_node:
            return GROOTLeaf(json_node["nodeid"], json_node["leaf"], depth)
        return TreeNode(
            json_node["nodeid"],
            json_node["depth"],
            json_node["split"],
            json_node["split_condition"],
            json_node["yes"],
            json_node["no"],
            json_node["missing"],
            False,
            TreeNode.from_json(json_node["children"][0], depth + 1),
            TreeNode.from_json(json_node["children"][1], depth + 1)
        )


class Leaf(TreeNode):
    def __init__(self, node_id, probabilities, depth):
        self.probs = probabilities
        super().__init__(node_id, depth, None, None, None, None, None, True, None, None)

    def get_leaf_value(self, fs):
        return np.array(self.probs)

    def _branch(self, indent_branches: bool):
        indent = self.indent if indent_branches else ""
        indent_line = "\n" if indent_branches else " "

        line_count = 0
        probs_per_line = 6
        r = ""
        for i in range(self.num_classes):
            r += f"{indent}res[{i}] = {self.probs[i]};{indent_line}"
            line_count += 1
            if line_count % probs_per_line == 0:
                r += "\n"
        return r+f"{indent}return res;"

    @property
    def num_classes(self):
        return len(self.probs)


class GROOTLeaf(Leaf):
    def __init__(self, node_id, proba_1, depth):
        probs = [1-proba_1, proba_1]
        super().__init__(node_id, probs, depth)


def visualise(clf, feature_names, class_names, image_name):
    image_name = join(IMAGE_DIR, image_name)
    dot_data = StringIO()
    export_graphviz(clf,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_names,
                    class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(image_name)
    # Image(filename=image_name)


if __name__ == '__main__':
    from run_fate import generate_model_path
    _d = 'BC'
    _mt = 'GB'
    num_classes = get_num_classes(_d)
    c_names = [f'c{i}' for i in range(num_classes)]
    num_features = get_num_features(_d)
    f_names = [f'f{i}' for i in range(num_features)]
    clf = load(generate_model_path(_d, _mt))
    for estimator_id, estimators in enumerate(clf.estimators_):
        if _mt == 'GB':
            for i, e in enumerate(estimators):
                visualise(e, f_names, c_names,
                          f"{_mt}-{_d}_{estimator_id}-{i}.png")
        else:
            visualise(estimators, f_names, c_names, f"{_mt}-{_d}_{estimator_id}.png")
