import argparse
import pathlib
import shutil
import statistics
import subprocess
import time
from os.path import join

import constants
from constants import RESULTS_DIR, DISTANCE_NORMS, CONSISTENT_DRAWS, NUM_THREADS, MODEL_TYPES, DISTANCE_NORM, \
    NUM_ADV_CHECKS
from datasets import get_epsilon
from result import generate_result_filename, stats
from run_fate import load_model_params, load_execution_details, get_execution_time, \
    RawTextArgumentDefaultsHelpFormatter, load_num_checks
from utils import write_file

DATASETS_TO_RUN = ['BC', 'DIABETES', 'IJ', 'COV', 'HIGGS', '26', 'MNIST', 'FMNIST']
# DATASETS_TO_RUN = ['26', 'MNIST', 'FMNIST']
# DATASETS_TO_RUN = ['MNIST']
# DATASETS_TO_RUN = ['BC', 'DIABETES', 'IJ', 'COV', 'HIGGS', '26']
# DATASETS_TO_RUN = ['26']
EPSILONS = [str(constants.DEFAULT_EPSILON)]
SKIP_COMPILE = True  # Skips compilation for run 2 ... REPETITIONS
# EPSILONS = ['0.2', '0.5', '1.0']
# EPSILONS = ['0.02']
cwd = pathlib.Path(__file__).parent.absolute()


def run_fuzzer():
    for dataset in DATASETS_TO_RUN:
        for epsilon in EPSILONS:
            for i in range(REPETITIONS):
                epsilon = epsilon if constants.FORCE_DEFAULT_EPSILON else get_epsilon(dataset)
                print(f'Running {dataset} {epsilon} {i+1}')
                command = ['python3', 'python/run_fate.py', dataset, MODEL_TYPE, str(epsilon), str(i)]

                if i != 0 and SKIP_COMPILE and constants.CONSISTENT_DRAWS:
                    command.append('-nc')
                if _args.quick:
                    command.append("-q")
                if VERBOSE:
                    process = subprocess.Popen(command, cwd=cwd.parent.absolute())
                else:
                    process = subprocess.Popen(command, cwd=cwd.parent.absolute(),
                                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    err_str = process.stderr.read()
                    if len(err_str) > 0:
                        print(err_str)
                process.wait()


def parse_results():
    result_dict = dict()
    for dataset_name in DATASETS_TO_RUN:
        dataset_dict = dict()
        t = get_execution_time(dataset_name, MODEL_TYPE)
        num_est, max_depth, max_leaves, num_mutations, accuracy = load_model_params(dataset_name, MODEL_TYPE)
        num_checks = load_num_checks(dataset_name)

        model_info = {
            'Dataset': dataset_name,
            'Num estimators': num_est,
            'Max depth': max_depth,
            'Max leaves': max_leaves,
            'Accuracy': accuracy,
            'Mutations': num_mutations,
            'Number of Victim points': num_checks,
            'Time per datapoint': t,
            'Number of threads': NUM_THREADS,
            'Repetitions': REPETITIONS,
        }
        dataset_dict['model_info'] = model_info

        for epsilon in EPSILONS:
            epsilon = epsilon if constants.FORCE_DEFAULT_EPSILON else get_epsilon(dataset_name)
            epsilon = str(epsilon)
            epsilon_dict = dict()
            norm_values = [[] for _ in range(len(DISTANCE_NORMS))]
            avg_times_fuzzing = []
            avg_total_times = []
            time_lookup = []
            time_compiling = []
            time_file_generation = []
            time_processing = []
            number_victims_attacked = []
            number_victims_not_attacked = []
            total_float_misc = []
            avg_covs = []
            avg_execs = []

            no_ae_found_counts = dict()
            for n in range(num_checks):
                no_ae_found_counts[n] = 0

            for it in range(REPETITIONS):
                try:
                    _, avg_time_fuzzing, combined_avg_time_fuzzing, load_or_train_time,\
                        lookup_creation_time, file_generation_time, compilation_time, fuzzing_time,\
                        other_time, total_time, avg_cov, avg_exec = load_execution_details(dataset_name, MODEL_TYPE,
                                                                                           epsilon, it)
                except FileNotFoundError:
                    continue

                actual_total_time = total_time - load_or_train_time
                del total_time  # to make sure it cannot be used anymore

                avg_covs.append(avg_cov)
                avg_execs.append(avg_exec)

                avg_times_fuzzing.append(combined_avg_time_fuzzing)
                time_lookup.append(lookup_creation_time)
                time_compiling.append(compilation_time)
                time_file_generation.append(file_generation_time)

                result_filename = join(RESULTS_DIR, generate_result_filename(dataset_name, epsilon, it))
                # print('Processing ', result_filename)
                try:
                    with open(result_filename, 'r') as file:
                        contents = file.readlines()
                    first_line_parts = contents[0].split(",")
                    num_attacked, num_no_attack, num_mis_classified, avg_best_norm, processing_time = \
                        int(first_line_parts[0]), int(first_line_parts[1]), \
                        int(first_line_parts[2]), float(first_line_parts[3]), float(first_line_parts[4])
                except FileNotFoundError:
                    print(f'WARNING!! File {result_filename} was not found')
                    continue

                if num_attacked == num_no_attack == 0:
                    continue

                time_processing.append(processing_time)
                number_victims_attacked.append(num_attacked)
                number_victims_not_attacked.append(num_no_attack)

                del avg_best_norm
                total_float_misc.append(num_mis_classified)

                total_points = num_attacked + num_no_attack
                avg_total_times.append(actual_total_time/total_points)

                if num_attacked == 0:
                    break

                for norm_index in range(len(DISTANCE_NORMS)):
                    line = contents[norm_index+1]
                    # Norm values for each datapoint for which an AE was found
                    norm_results = [float(e) for e in line.split(",")]
                    norm_values[norm_index].append(statistics.mean(norm_results))

                point_ids_no_ae_line = contents[len(DISTANCE_NORMS)+1]
                if point_ids_no_ae_line == "" or point_ids_no_ae_line == "\n":
                    point_ids_no_ae = []
                else:
                    point_ids_no_ae = [int(i) for i in point_ids_no_ae_line.split(",")]
                if CONSISTENT_DRAWS:
                    for pid in point_ids_no_ae:
                        no_ae_found_counts[pid] += 1

            for norm_id, norm in enumerate(DISTANCE_NORMS):
                epsilon_dict[generate_norm_title(norm)] = norm_values[norm_id]
            epsilon_dict['Avg time Fuzzing'] = avg_times_fuzzing
            epsilon_dict['Avg total time'] = avg_total_times
            epsilon_dict['Time lookup init'] = time_lookup
            epsilon_dict['Time compilation'] = time_compiling
            epsilon_dict['Time file generation'] = time_file_generation
            epsilon_dict['Time processing'] = time_processing
            epsilon_dict['Number of victims attacked'] = number_victims_attacked
            epsilon_dict['Number of victims not attacked'] = number_victims_not_attacked
            epsilon_dict['Total float mis-classification'] = total_float_misc
            epsilon_dict['Average Coverage'] = avg_covs
            epsilon_dict['Average Executions per Second'] = avg_execs

            if CONSISTENT_DRAWS:
                no_ae_found_counts_list = [e for e in no_ae_found_counts.values() if e != 0]
            else:
                no_ae_found_counts_list = []
            epsilon_dict['Not attacked victims counts'] = no_ae_found_counts_list

            dataset_dict[epsilon] = epsilon_dict
        result_dict[dataset_name] = dataset_dict
    return result_dict


def generate_norm_title(norm):
    return f'Avg best examples {norm}'


def to_csv(filename, result_dict):
    output_str = ""
    for dataset_name in DATASETS_TO_RUN:
        dataset_results = result_dict[dataset_name]
        model_info = dataset_results['model_info']
        output_str += ",".join([str(k) for k in model_info.keys()]) + "\n"
        output_str += ",".join([str(v) for v in model_info.values()]) + "\n"
        for epsilon in EPSILONS:
            epsilon = epsilon if constants.FORCE_DEFAULT_EPSILON else get_epsilon(dataset_name)
            epsilon = str(epsilon)
            epsilon_results = dataset_results[epsilon]
            output_str += f"Epsilon,{epsilon}\n"
            output_str += f",Min,Max,Mean,Median,Std\n"
            output_str += "".join([_stats_string(k, v) for k, v in epsilon_results.items()])
            output_str += "\n"
        output_str += "\n"

    write_file(filename, output_str)


def short_output(filename, result_dict):
    out_str = "dataset,epsilon,avg norm,avg time,avg n\n"
    for dataset in DATASETS_TO_RUN:
        for epsilon in EPSILONS:
            epsilon = epsilon if constants.FORCE_DEFAULT_EPSILON else get_epsilon(dataset)
            epsilon = str(epsilon)
            _, _, avg_best, _, _ = stats(result_dict[dataset][epsilon][generate_norm_title(DISTANCE_NORM)])
            _, _, avg_n, _, _ = stats(result_dict[dataset][epsilon]['Number of victims attacked'])
            _, _, avg_time, _, _ = stats(result_dict[dataset][epsilon]['Avg time Fuzzing'])
            print(f'{dataset} - {epsilon}: norm = {avg_best}; time = {avg_time}; n = {avg_n}')
            out_str += f'{dataset},{epsilon},{avg_best},{avg_time},{avg_n}\n'
    write_file(filename, out_str)


def _stats_string(name, vals):
    res = f"{name},"
    res += ",".join([str(e) for e in stats(vals)])
    return res + "\n"


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description="Run FATE for multiple datasets and epsilons",
                                      formatter_class=RawTextArgumentDefaultsHelpFormatter)
    _parser.add_argument('model_type', type=str, choices=MODEL_TYPES, default='GB', nargs='?',
                         help="the type of the model.\noptions: " + ", ".join(MODEL_TYPES),
                         metavar="model_type")
    _parser.add_argument('repetitions', type=int, default=5, nargs='?',
                         help="the number of repetitions for an experiment")
    _parser.add_argument('output_file', type=str, default='results', nargs='?',
                         help="the name of the output file without extension (will be .csv)")
    _parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
                         help="show output of each experiment")
    _parser.add_argument('-q', '--quick', dest='quick', action='store_true', default=False,
                         help=f"Execute for {constants.NUM_ADV_QUICK} instead of {NUM_ADV_CHECKS} victims")
    _parser.add_argument('-po', '--parse_only', dest='parse_only', action='store_true', default=False,
                         help="only parse results from a previous run")
    _args = _parser.parse_args()

    MODEL_TYPE = _args.model_type
    REPETITIONS = _args.repetitions
    VERBOSE = _args.verbose

    quick_str = "_quick" if _args.quick else ""

    _filename_base = _args.output_file + f'_{MODEL_TYPE}'
    _filename_base_short = _filename_base + quick_str + "_short"
    _filename = _filename_base + quick_str + ".csv"
    _filename_short = _filename_base_short + ".csv"
    _filename_settings = _filename_base + quick_str + "_settings.py"

    settings_path = join("python", "constants.py")
    shutil.copy(settings_path, _filename_settings)

    if not _args.parse_only:
        t_0 = time.time()
        run_fuzzer()
        print(f'Running all took {time.time() - t_0} seconds')

    results = parse_results()
    short_output(_filename_short, results)
    to_csv(_filename, results)
