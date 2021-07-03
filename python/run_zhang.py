import json
import subprocess
from os.path import join

from constants import DISTANCE_NORM, NUM_THREADS, ZHANG_CONFIG_DIR, ZHANG_DATA_DIR, \
    JSON_DIR, get_num_adv
from datasets import get_num_classes, get_num_features
from result import stats
from run_fate import generate_model_path, generate_x_test_path, \
    generate_y_test_path, take_n
from utils import load, write_file, makedir, to_json

# _datasets = ['COV-M']
# _datasets = ['BC', 'DIABETES', 'IJ', 'COV', 'HIGGS', '26', 'MNIST', 'FMNIST']
# _datasets = ['WEB', 'VOWEL']
_datasets = ['MNIST', 'FMNIST']
_mts = ['GB', 'RF']
_num_iter = 5
_precision = 8
_output_filename = 'zhang_results.csv'


def _generate_json_filename(dataset_name, mt):
    return join(JSON_DIR, f'{dataset_name}_{mt}.json')


def _generate_zhang_data_filename(dataset_name, mt):
    return join(ZHANG_DATA_DIR, f'zhang_{dataset_name}_{mt}.test')


def _generate_zhang_config_filename(dataset_name, mt):
    return join(ZHANG_CONFIG_DIR, f'config_{dataset_name}_{mt}.json')


def _convert_to_zhang_input(x_test, y_test, filename):
    out_str = ""
    for label, features in zip(y_test, x_test):
        line = f"{label} "
        f_str = []
        for ii, f in enumerate(features):
            f_str.append(f"{ii}:{round(f, _precision)}")
        line += " ".join(f_str)
        out_str += line + "\n"
    write_file(filename, out_str)


def _create_config(dataset_name, mt, json_filename, data_filename, num_points):
    if DISTANCE_NORM == 'l_inf':
        nt = -1
    elif DISTANCE_NORM == 'l_1':
        nt = 1
    elif DISTANCE_NORM == 'l_2':
        nt = 2
    else:
        raise ValueError('Distance norm not recognised when creating config for Zhang')

    d = {
        "num_threads": NUM_THREADS,
        "enable_early_return": True,
        "inputs": data_filename,
        "model": json_filename,
        "num_classes": get_num_classes(dataset_name),
        "num_features": get_num_features(dataset_name),
        "feature_start": 0,
        "num_point": num_points,
        "num_attack_per_point": NUM_THREADS,
        "norm_type": nt,
        "search_mode": "lt-attack",
    }
    content = json.dumps(d)
    write_file(_generate_zhang_config_filename(dataset_name, mt), content)


def prepare_for_dataset(dataset_name, mt):
    makedir(ZHANG_CONFIG_DIR)
    classifier = load(generate_model_path(dataset_name, mt))

    x_test = load(generate_x_test_path(dataset_name, mt))
    y_test = load(generate_y_test_path(dataset_name, mt))
    # x_test, y_test = take_n(classifier, x_test, y_test, 50)
    x_test, y_test = take_n(classifier, x_test, y_test, get_num_adv())
    num_points = len(y_test)
    data_filename = _generate_zhang_data_filename(dataset_name, mt)
    _convert_to_zhang_input(x_test, y_test, data_filename)

    json_filename = _generate_json_filename(dataset_name, mt)
    to_json(classifier, json_filename, mt)

    _create_config(dataset_name, mt, json_filename, data_filename, num_points)
    return num_points


def run_for_dataset(dataset_name, mt):
    command = ['./lt_attack', _generate_zhang_config_filename(dataset_name, mt)]
    output = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    # print("output:", output)
    try:
        res = _parse_output(output)
    except IndexError as e:
        print('Error for output: ', output)
        print(e)
        raise IndexError(e)
    return res


def _parse_output(s: str):
    """
    Sample input string:
    Results for config:configs/breast_cancer_unrobust_20x500_norm2_lt-attack.json
    Average Norms: Norm(-1)=0.235932 Norm(1)=0.369484 Norm(2)=0.282763
    --- Timing Metrics ---
    |collect_histogram| disabled
    ## Number of float misclassifications:1
    ## Actual Examples Tested:496
    ## Time per point: 0.00643812

    output for DISTANCE_NORM == 'l_inf': 0.235932, 0.00643812
    """
    lines = s.splitlines()
    relevant_lines = lines[-7:]
    norms_entries = relevant_lines[1].split(" ")

    if DISTANCE_NORM == 'l_inf':
        avg_norm = float(norms_entries[2].split("=")[1])
    elif DISTANCE_NORM == 'l_1':
        avg_norm = float(norms_entries[3].split("=")[1])
    elif DISTANCE_NORM == 'l_2':
        avg_norm = float(norms_entries[4].split("=")[1])
    else:
        raise ValueError('Unexpected norm when running Zhang')

    time_per_point = float(relevant_lines[-1].split(" ")[-1])
    num_examples = int(relevant_lines[-2].split(":")[-1])
    float_misc = int(relevant_lines[-3].split(":")[-1])
    return avg_norm, time_per_point, num_examples, float_misc


def run_all():
    out_str = "Model Type,Dataset," \
              "Min - norm,Max - norm,Mean - norm,Median - norm,Std - norm," \
              "Min - time,Max - time,Mean - time,Median - time,Std - time," \
              "Min - num-attacked,Max - num-attacked,Mean - num-attacked,Median - num-attacked,Std - num-attacked," \
              "Num examples tested,Avg float mis-classifications\n"
    for dataset in _datasets:
        for mt in _mts:
            actual_num_points = prepare_for_dataset(dataset, mt)
            try:
                avg_norms = []
                avg_times = []
                avg_examples = []
                avg_float_misc = 0
                for i in range(_num_iter):
                    try:
                        avg_norm, time_per_point, num_examples, float_misc = run_for_dataset(dataset, mt)
                        avg_norms.append(avg_norm)
                        avg_times.append(time_per_point)
                        avg_examples.append(num_examples)
                        perc_str = f'({round(num_examples/actual_num_points*100, 2)}%)'
                        print(f'{mt}-{dataset}: avg-norm = {avg_norm}, avg-time = {time_per_point}, '
                              f'num-attacked: {num_examples}/{actual_num_points} {perc_str}, '
                              f'float mis-classifications: {float_misc}')
                        avg_float_misc += float_misc
                    except Exception as e:
                        # For RF-MNIST sometimes the output is different, which makes
                        # avg_norm = float(norms_entries[2].split("=")[1]) fail.
                        # Just continue, some runs will succeed
                        print(e)
                out_str += f'{mt},{dataset},' + ",".join([str(e) for e in stats(avg_norms)]) + \
                           "," + ",".join([str(e) for e in stats(avg_times)]) + \
                           "," + ",".join([str(e) for e in stats(avg_examples)]) + f",{actual_num_points}" \
                                                                                   f",{avg_float_misc/_num_iter}\n"
            except (ValueError, FileNotFoundError):
                print("ERROR run was not possible for", dataset, mt)
            print()
    write_file(_output_filename, out_str)


if __name__ == '__main__':
    run_all()
