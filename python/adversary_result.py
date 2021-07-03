import math
from os import listdir
from os.path import isfile, join
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constants import SMALL_PERTURBATION_THRESHOLD


class AdversaryResult:
    def __init__(self, filename, fuzzed_features, fuzzed_class, original_features, original_class, check_num,
                 fuzzer_probs):
        self.filename = filename
        self.fuzzed_features = fuzzed_features
        self.fuzzed_class = fuzzed_class
        self.original_features = original_features
        self.original_class = original_class
        self.check_num = check_num
        self.fuzzer_probs = fuzzer_probs
        self.valid = True

    @property
    def key(self):
        return f'{self.original_class}->{self.fuzzed_class}'

    def diffs(self):
        return np.abs(self.fuzzed_features - self.original_features)

    @staticmethod
    def differences(a, b):
        return np.abs(b - a)

    def avg_diff(self):
        return np.average(self.diffs())

    def big_diffs(self):
        return [e for e in self.diffs() if e > SMALL_PERTURBATION_THRESHOLD]

    def avg_diff_big(self):
        return np.average(self.big_diffs())

    def dist(self):
        return math.dist(self.fuzzed_features, self.original_features)

    def l_0_big(self):
        return len(self.big_diffs())

    def l_0(self):
        # Number of elements that is different in a vector / non-zero elements.
        return len([e for i, e in enumerate(self.fuzzed_features) if e != self.original_features[i]])

    def l_1(self):
        # Sum of absolute difference
        return sum(self.diffs())

    def l_2(self):
        # Euclidean distance
        return self.dist()

    def l_inf(self):
        # The largest (difference) of any element of the vectors
        return max(self.diffs().tolist())

    def l_inf_other(self, other):
        return max(self.differences(self.original_features, other))

    def result_for_norm(self, norm):
        return eval("self."+norm+"()")

    def print_norms(self):
        print('l_0:   ', self.l_0())
        print('l_0_b: ', self.l_0_big())
        print('l_1:   ', self.l_1())
        print('l_2:   ', self.l_2())
        print('l_inf: ', self.l_inf())

    def visualise_image(self):
        plt.title(f'Original class is {self.original_class}, adversary class is {self.fuzzed_class}')
        plt.imshow(self.pixels(self.fuzzed_features), cmap='gray')
        plt.show()

    def image_compare(self):
        fig = plt.figure()
        fig.suptitle(f'Original Class is {self.original_class}, adversary class is {self.fuzzed_class}')

        fig.add_subplot(1, 2, 1)  # Left
        plt.imshow(self.pixels(self.fuzzed_features), cmap='gray')

        fig.add_subplot(1, 2, 2)  # Right
        plt.imshow(self.pixels(self.original_features), cmap='gray')

        plt.show()

    def investigate(self, actual_model, predicted_probs, fatal=True):
        print('ERROR for file: ', self.filename)
        print('Generated adv example: ', self.fuzzed_features)
        print('Fuzzer probs: ', self.fuzzer_probs)
        print('Model probs: ', predicted_probs)
        print('Diff: ', self.fuzzer_probs-predicted_probs)
        print('Avg diff: ', np.average(self.fuzzer_probs-predicted_probs))
        print('Max diff: ', np.max(self.fuzzer_probs-predicted_probs))

        err_str = (f"The trained model predicted {actual_model}, but the fuzzer predicts {self.fuzzed_class}. "
                   f"Skipping adv example.")
        if fatal:
            raise ValueError(err_str)
        print(err_str)

    @staticmethod
    def pixels(features):
        return features.reshape((28, 28))

    @staticmethod
    def parse_results(adv_path, original_fs: dict, num_features: int) -> List['AdversaryResult']:
        """
        original_fs is a dict[int(check_num) -> original_features]
        """

        adv_files = [join(adv_path, f) for f in listdir(adv_path) if isfile(join(adv_path, f)) and 'probasmall' not in f]
        print(f"{len(adv_files)} adversarial examples were found.")

        adversary_results = []
        for adv_filename in adv_files:
            with open(adv_filename, 'r') as file:
                contents = file.read()
            try:
                # Try block for the case that two processes were writing to the same file

                splitted = contents.split(',')

                fuzzed_features = np.array([float(e) for e in splitted[:num_features]], dtype='float32')
                verify(fuzzed_features)  # may raise ValueError

                original_class = int(splitted[num_features])
                fuzzed_class = int(splitted[num_features+1])
                check_num = int(splitted[num_features+2])

                original_features = np.array(original_fs[check_num], dtype='float32')
                fuzzer_probs = np.array([float(e) for e in splitted[num_features+3:]])  # Fuzzer probabilities
                verify(fuzzer_probs)  # may raise ValueError

                adversary_results.append(
                    AdversaryResult(adv_filename, fuzzed_features, fuzzed_class, original_features, original_class,
                                    check_num, fuzzer_probs)
                )
            except (ValueError, IndexError):
                continue

        if len(adv_files) != len(adversary_results):
            print(f'Warning!! {len(adv_files) - len(adversary_results)} files could not be parsed')

        return adversary_results


def verify(features):
    """
    Verifies that all features are finite i.e. do not contain NaN, inf, -inf values.
    """
    if not np.isfinite(features).all():
        raise ValueError
