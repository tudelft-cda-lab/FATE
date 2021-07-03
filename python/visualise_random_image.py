from adversary_result import AdversaryResult
from constants import ADV_DIR
from datasets import get_num_features
from generate_code import read_check_files
from random import randrange

CHOOSE_RANDOM = True


def run(adv_path, num_features):
    adv_results = AdversaryResult.parse_results(adv_path, read_check_files(), num_features)

    if CHOOSE_RANDOM:
        num_adv = len(adv_results)
        index = randrange(num_adv)
        best_result = adv_results[index]
    else:
        best_result = adv_results[0]
        for result in adv_results:
            if result.l_inf() < best_result.l_inf():
                best_result = result

    # print('Diff per non-0 feature: ', best_result.big_diffs())
    # print('Avg diff: ', best_result.avg_diff())
    # print('Avg diff non-zero: ', best_result.avg_diff_big())
    # print('Euclidean distance: ', best_result.dist())

    best_result.print_norms()
    best_result.image_compare()


if __name__ == '__main__':
    dataset = 'FMNIST'
    run(ADV_DIR, get_num_features(dataset))
