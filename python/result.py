import statistics
import time
from os import listdir
from os.path import join, isfile
from typing import List

import numpy as np

from adversary_result import AdversaryResult
from constants import RESULTS_DIR, ADV_DIR, DISTANCE_NORMS, DISTANCE_NORM, ALLOW_FLOAT_MIS_CLASSIFICATION, \
    FAILURE_THRES, FILTER_BAD_AE
from generate_code import read_check_files
from run_fate import generate_model_path
from utils import makedir, load


def find_ids_no_ae(num_points):
    adv_files = [join(ADV_DIR, f) for f in listdir(ADV_DIR) if isfile(join(ADV_DIR, f)) and "probasmall" not in f]
    check_nums_found = set()
    for filepath in adv_files:
        filename = filepath.replace(ADV_DIR, "")
        check_part = filename.split("-")[0]
        check_num = int(check_part.split("_")[1])
        check_nums_found.add(check_num)
    return [i for i in range(num_points) if i not in check_nums_found]


def process_adversarial_examples(dataset_name, model_type, epsilon, num_features, iteration, num_points,
                                 ae_lookup=None):
    t_0 = time.time()

    adversary_results = AdversaryResult.parse_results(ADV_DIR, read_check_files(), num_features)
    original_num_ae = len(adversary_results)
    adversary_results = _verify_adversarial_example_files(dataset_name, model_type, adversary_results)
    count_float_mis_classification = original_num_ae - len(adversary_results)
    count_valid_examples = len(adversary_results)
    if count_valid_examples == 0:
        _output(dataset_name, epsilon, 0, 0, None, None, None, None, None, iteration, 0)
        return 0, 1, 0, 0, 0

    best_results = _best_results_per_datapoint(adversary_results)
    
    percentage_str = str(
        round(count_float_mis_classification/(count_float_mis_classification+count_valid_examples)*100, 2)
    )+"%" if len(adversary_results) != 0 else "-"
    num_data_points_attacked = len(set([result.check_num for result in adversary_results]))
    num_data_points_attacked_valid = len(set([r.check_num for r in adversary_results
                                              if not FILTER_BAD_AE or r.result_for_norm('l_inf') < FAILURE_THRES]))

    avg_best_norm = round(float(np.average(
        [r.result_for_norm(DISTANCE_NORM) for r in best_results[DISTANCE_NORM].values()]
    )), 4)

    l_0 = [r.result_for_norm('l_0') for r in best_results['l_0'].values()]
    l_1 = [r.result_for_norm('l_1') for r in best_results['l_1'].values()]
    l_2 = [r.result_for_norm('l_2') for r in best_results['l_2'].values()]
    l_inf = [r.result_for_norm('l_inf') for r in best_results['l_inf'].values()]

    point_ids_no_ae = [i for i in range(num_points) if i not in best_results[DISTANCE_NORM]] \
        if num_points is not None else None

    print(f'Of which {count_valid_examples} examples were valid.')
    print(f'And {count_float_mis_classification} ({percentage_str}) floating point mis-classifications were skipped')
    print()
    datapoint_perc_string = "" if num_points is None else \
        f" out of {num_points}({round(num_data_points_attacked/num_points*100, 2)}%)"
    print(f'{num_data_points_attacked}{datapoint_perc_string} data points were attacked')
    print(f'Of which {num_data_points_attacked_valid} were left after float thres')
    print(f'With average best l-inf norm: {avg_best_norm}')
    print()

    # flips, counts_original_class, counts_fuzzed_class = _adv_class_changes(adversary_results)
    # print('Class flips: ', flips)
    # print('How often a certain class was adversarially changed: ', counts_original_class)
    # print('How often a class was adversarially predicted: ', counts_fuzzed_class)

    stats(l_0, 'l_0  ', True)
    stats(l_1, 'l_1  ', True)
    stats(l_2, 'l_2  ', True)
    stats(l_inf, 'l_inf', True)

    processing_time = round(time.time() - t_0, 4)

    _output(dataset_name, epsilon, count_float_mis_classification, avg_best_norm,
            l_0, l_1, l_2, l_inf, point_ids_no_ae, iteration, processing_time)

    avg_dist_to_own_class = 0
    avg_dist_to_other_class = 0
    if ae_lookup is not None:
        count = 0
        for r in best_results[DISTANCE_NORM].values():
            dist_to_other_index = ae_lookup[r.original_class]["ann"].get_nns_by_vector(r.fuzzed_features, 1)[0]
            instances = ae_lookup[r.original_class]["instances"]
            closest = instances[dist_to_other_index]
            dist_to_other = r.l_inf_other(np.array(closest))
            avg_dist_to_other_class += dist_to_other
            if len(ae_lookup) == 2:
                # Binary classification

                dist_to_own_index = ae_lookup[abs(r.original_class-1)]["ann"].get_nns_by_vector(r.fuzzed_features, 1)[0]
                instances = ae_lookup[abs(r.original_class-1)]["instances"]
                closest = instances[dist_to_own_index]
                dist_to_own = r.l_inf_other(np.array(closest))

                # print(dist_to_own)
                avg_dist_to_own_class += dist_to_own
            count += 1
        avg_dist_to_other_class /= count
        avg_dist_to_own_class /= count

    return processing_time, avg_best_norm, num_data_points_attacked_valid, avg_dist_to_own_class, \
        avg_dist_to_other_class


def _verify_adversarial_example_files(dataset_name, model_type, adversary_results):
    if len(adversary_results) == 0:
        return []
    model_path = generate_model_path(dataset_name, model_type)
    model = load(model_path)
    # An exception will be raised in predict if the input contains NaN,
    # infinity or a value too large for float32
    predictions = model.predict([adv_res.fuzzed_features for adv_res in adversary_results])

    for i, result in enumerate(adversary_results):
        predicted_class_by_model = predictions[i]
        if result.fuzzed_class != predicted_class_by_model:
            # The fuzzer and model disagree on the prediction
            # if result.original_class != predicted_class_by_model then the fuzzer has still
            # found an adv example, even though fuzzer and model disagree.
            if result.original_class == predicted_class_by_model:
                # This is not actually an adversarial example.
                result.valid = ALLOW_FLOAT_MIS_CLASSIFICATION

    return [ar for ar in adversary_results if ar.valid]


def _adv_class_changes(adversary_results):
    flips = dict()
    counts_original_class = dict()
    counts_fuzzed_class = dict()
    for result in adversary_results:
        if result.key not in flips:
            flips.update({result.key: 0})
        flips[result.key] += 1

        if result.original_class not in counts_original_class:
            counts_original_class.update({result.original_class: 0})
        counts_original_class[result.original_class] += 1

        if result.fuzzed_class not in counts_fuzzed_class:
            counts_fuzzed_class.update({result.fuzzed_class: 0})
        counts_fuzzed_class[result.fuzzed_class] += 1

    return flips, counts_original_class, counts_fuzzed_class


def _best_results_per_datapoint(adversary_results):
    """
    output: {
        'l_inf' {
            0: AdversaryResult,
            1: AdversaryResult,
            ...
        },
        'l_0': {...},
        'l_1': {...},
        'l_2': {...},
    }
    """
    best_results = dict()
    for norm in DISTANCE_NORMS:
        best_results.update({norm: dict()})
        best_result_per_norm = best_results[norm]

        for result in adversary_results:
            data_point_id = result.check_num
            if FILTER_BAD_AE and result.result_for_norm('l_inf') > FAILURE_THRES:
                continue
            if data_point_id not in best_result_per_norm:
                best_result_per_norm.update({data_point_id: result})
            else:
                result_norm = result.result_for_norm(norm)
                current_best = best_result_per_norm[data_point_id].result_for_norm(norm)
                if result_norm < current_best:
                    best_result_per_norm[data_point_id] = result
    return best_results


def stats(l_norm_list: List[float], name=None, stdout=False):
    if len(l_norm_list) == 0:
        return None, None, None, None, None

    precision = 4
    minn = np.round(np.min(l_norm_list), precision)
    maxx = np.round(np.max(l_norm_list), precision)
    avg = np.round(np.average(l_norm_list), precision)
    median = round(statistics.median(l_norm_list), precision)
    if len(l_norm_list) > 1:
        std_dev = round(statistics.stdev(l_norm_list, avg), precision)
    else:
        std_dev = 0
    
    if stdout:
        print(f'{name} - min: {minn}, max: {maxx}, avg: {avg}, median: {median}, std: {std_dev}')
    return minn, maxx, avg, median, std_dev


def _output(dataset_name, epsilon, num_mis_classified, avg_best_norm,
            l0, l1, l2, linf, point_ids_no_ae, run_i, processing_time):
    makedir(RESULTS_DIR)
    output_file_name = join(RESULTS_DIR, generate_result_filename(dataset_name, epsilon, run_i))
    num_attacked = len(linf) if linf is not None else 0
    num_no_attack = len(point_ids_no_ae) if point_ids_no_ae is not None else 0
    if num_attacked == 0:
        output_str = "0,0,0,0,0"
    else:
        output_str = f"{num_attacked},{num_no_attack},{num_mis_classified},{avg_best_norm},{processing_time}\n"
        for l_norm in [l0, l1, l2, linf]:
            output_str += ",".join([str(norm) for norm in l_norm]) + "\n"
        if point_ids_no_ae is not None:
            output_str += ",".join([str(i) for i in point_ids_no_ae])
        output_str += "\n"
    with open(output_file_name, 'w+') as file:
        file.write(output_str)


def generate_result_filename(dataset_name, epsilon, run_i):
    return f'{dataset_name}-{epsilon}-{run_i}.csv'
