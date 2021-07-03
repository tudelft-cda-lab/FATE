import statistics
import subprocess
import argparse
import os
import random
import struct
import time
from math import ceil
from multiprocessing import Process, Queue, Manager
from collections import Counter
from os.path import join
from queue import Empty
from annoy import AnnoyIndex

import numpy as np

import constants
from constants import RESULTS_DIR, CORPUS_DIR, NUM_FEATURES_PATH, MODEL_TYPES, \
    DEFAULT_LEARNING_RATE_GB, FUZZ_ONE_POINT_PER_INSTANCE, ADV_DIR, BYTES_PER_FEATURE, \
    MODEL_DIR, SHOW_OUTPUT, NUM_THREADS, OUTPUT_FILE, CONSISTENT_DRAWS, \
    INITIALIZE_WITH_AE, K_ANN, DOUBLE_FUZZ_WITH_AE, ANN_TREES, TIME_PRECISION, \
    INVESTIGATE, INVESTIGATE_WITH_SCATTER, COVERAGES_DIR, FUZZ_WITHOUT_COVERAGE_GUIDANCE, \
    INITIALIZE_WITH_FULL_TRAIN_SET, CREATE_LOOKUP, NUM_INVESTIGATE_RUNS, MAX_POINTS_LOOKUP, MUTATE_DEPTH, \
    get_num_adv, NUM_ADV_CHECKS
from forest import RandomForest, GradientBoostingForest, BaseForest
from datasets import MODEL_SETTINGS, get_execution_time, DATASETS, get_epsilon, get_entropic
from plot_coverages import plot_lines, plot_scatter
from utils import load, save, make_or_empty_dir, write_file, remove_file


def _n_indices(num_choice, n):
    all_indices = [i for i in range(num_choice)]
    if CONSISTENT_DRAWS:
        random.seed(0)
    return random.sample(all_indices, k=n)  # Draw without repetition


def take_n(clf, x_test, y_test, n):
    """
    clf can either be a Forest or SKLEARN Classifier
    """
    actual_outcomes = clf.predict(x_test)
    allowed_x = []
    allowed_y = []
    for i, expected_outcome in enumerate(y_test):
        actual_outcome = actual_outcomes[i]

        if expected_outcome == actual_outcome:
            allowed_x.append(x_test[i])
            allowed_y.append(expected_outcome)

    if n >= len(allowed_x):
        return allowed_x, allowed_y

    indices = _n_indices(len(allowed_y), n)
    return [allowed_x[i] for i in indices], [allowed_y[i] for i in indices]


def _check_missing_values(xs, filter_invalid=False):
    res = []
    expected_length = len(xs[0])
    for i, fs in enumerate(xs):
        if len(fs) != expected_length:
            if filter_invalid:
                continue
            raise ValueError(f'Number of features for row {i} (0-indexed) is {len(fs)}, '
                             f'but expected {expected_length}.')
        else:
            res.append(fs)
    return res


def generate_model_path(dataset_name, mt):
    return join(MODEL_DIR, f"model_{mt}_{dataset_name}.sav")


def generate_x_test_path(dataset_name, mt):
    return join(MODEL_DIR, f"model_{mt}_{dataset_name}_x_test.sav")


def generate_y_test_path(dataset_name, mt):
    return join(MODEL_DIR, f"model_{mt}_{dataset_name}_y_test.sav")


def generate_x_train_path(dataset_name, mt):
    return join(MODEL_DIR, f"model_{mt}_{dataset_name}_x_train.sav")


def generate_y_train_path(dataset_name, mt):
    return join(MODEL_DIR, f"model_{mt}_{dataset_name}_y_train.sav")


def generate_model_params_path(dataset_name, mt):
    return join(RESULTS_DIR, f"{mt}_{dataset_name}.csv")


def generate_execution_details_path(dataset_name, mt, epsilon, iteration):
    return join(RESULTS_DIR, f"{mt}_{dataset_name}_execution_details_{epsilon}_{iteration}.csv")


def generate_num_checks_path(dataset_name):
    return join(RESULTS_DIR, f"{dataset_name}_num_checks.txt")


def generate_coverages_path(dataset_name, epsilon, mt, scatter):
    return join(COVERAGES_DIR, f'coverages_{mt}_{dataset_name}_{epsilon}_{scatter}.csv')


def _verify_class_labels(ys, num_classes):
    for i, y in enumerate(ys):
        if int(y) != y:
            raise ValueError(f'Label of datapoint {i} is not an int')
        if y >= num_classes:
            raise ValueError(f'Label of datapoint {i} ({y}) bigger or equal to the number of classes ({num_classes})')
        if y < 0:
            raise ValueError(f'Label of datapoint {i} ({y}) below 0')


def write_num_checks(dataset_name, num_checks):
    with open(generate_num_checks_path(dataset_name), 'w+') as file:
        file.write(str(num_checks))


def load_num_checks(dataset_name):
    with open(generate_num_checks_path(dataset_name), 'r') as file:
        contents = file.read()
    return int(contents)


def save_model_params(dataset_name, num_est, max_depth, max_leaves, num_mutations, accuracy):
    params = [str(e) for e in [num_est, max_depth, max_leaves, num_mutations, accuracy]]
    with open(generate_model_params_path(dataset_name, MODEL_TYPE), 'w+') as file:
        file.write(",".join(params))


def load_model_params(dataset_name, mt, base_dir=""):
    with open(join(base_dir, generate_model_params_path(dataset_name, mt)), 'r') as file:
        contents = file.readline()
    fields = contents.split(",")
    num_est = fields[0]
    max_depth = fields[1]
    max_leaves = fields[2]
    num_mutations = fields[3]
    accuracy = fields[4]
    return num_est, max_depth, max_leaves, num_mutations, accuracy


def write_execution_details(dataset_name, num_points, avg_time_fuzzing, combined_avg_time_fuzzing, load_or_train_time,
                            lookup_creation_time, file_generation_time, compilation_time, fuzzing_time,
                            other_time, total_time, coverage, executions_per_s):
    contents = f'{num_points},{avg_time_fuzzing},{combined_avg_time_fuzzing},{load_or_train_time},' \
               f'{lookup_creation_time},{file_generation_time},{compilation_time},{fuzzing_time},' \
               f'{other_time},{total_time},{coverage},{executions_per_s}'
    with open(generate_execution_details_path(dataset_name, MODEL_TYPE, _eps, ITERATION), 'w+') as file:
        file.write(contents)


def load_execution_details(dataset_name, mt, epsilon, iteration):
    with open(generate_execution_details_path(dataset_name, mt, epsilon, iteration), 'r') as file:
        contents = file.read()
    splitted = contents.split(",")
    return int(splitted[0]), *[float(splitted[i]) for i in range(1, 12)]


def _write_num_features(num_features: int):
    # For communication with the fuzzer (multiplied by bytes_per_feature for number of input bytes)
    with open(NUM_FEATURES_PATH, 'w+') as file:
        file.write(str(num_features))


def read_num_features() -> int:
    with open(NUM_FEATURES_PATH, 'r') as file:
        contents = file.read()
    return int(contents)


def load_data(dataset_name):
    print("Loading data...")
    try:
        data_func = MODEL_SETTINGS[dataset_name]["data"]
        x_train, x_test, y_train, y_test = data_func()
        print("Loaded data")
        return x_train, x_test, y_train, y_test
    except KeyError:
        raise KeyError(f'{dataset_name} is not a recognised dataset')


def generate_model_and_run(dataset_name, epsilon, from_json=False, json_filename=None,
                           learning_rate=DEFAULT_LEARNING_RATE_GB):
    start_0 = time.time()
    num_mutations = 1  # Artifact, can/should be removed

    if from_json:
        x_train, x_test, y_train, y_test = load_data(dataset_name)
        x_test = _check_missing_values(x_test)  # May raise error.
        num_classes = len(set(y_test))
        _verify_class_labels(y_test, num_classes)  # May raise error.
        num_features = len(x_test[0])

        if json_filename is None:
            raise ValueError('Please provide a JSON filename to load from')
        if MODEL_TYPE != 'RF':
            raise NotImplementedError('We only support RF trees in JSON format')

        forest = RandomForest.from_file(json_filename, num_classes)
        classifier = forest
        save(forest, generate_model_path(dataset_name, MODEL_TYPE))
        save_model_params(dataset_name, forest.num_estimators, None, None, num_mutations, None)
    else:
        if RELOAD:
            x_train, x_test, y_train, y_test = load_data(dataset_name)
            x_train = _check_missing_values(x_train)  # May raise error.
            x_test = _check_missing_values(x_test)  # May raise error.
            num_classes = len(set(y_test))
            _verify_class_labels(y_train, num_classes)  # May raise error.
            _verify_class_labels(y_test, num_classes)  # May raise error.

            num_est = MODEL_SETTINGS[dataset_name][MODEL_TYPE]['num_est']
            max_depth = MODEL_SETTINGS[dataset_name][MODEL_TYPE]['max_depth']
            max_leaves = MODEL_SETTINGS[dataset_name][MODEL_TYPE]['max_leaves']
            target_acc = MODEL_SETTINGS[dataset_name][MODEL_TYPE]['target_acc']
            max_iter = MODEL_SETTINGS[dataset_name][MODEL_TYPE]['max_iter']

            classifier, accuracy = BaseForest.train(MODEL_TYPE, x_train, x_test, y_train, y_test, num_est,
                                                    max_depth, max_leaves, target_acc, max_iter, learning_rate)

            save(classifier, generate_model_path(dataset_name, MODEL_TYPE))
            save(x_train, generate_x_train_path(dataset_name, MODEL_TYPE))
            save(y_train, generate_y_train_path(dataset_name, MODEL_TYPE))
            save(x_test, generate_x_test_path(dataset_name, MODEL_TYPE))
            save(y_test, generate_y_test_path(dataset_name, MODEL_TYPE))
            save_model_params(dataset_name, num_est, max_depth, max_leaves, num_mutations, accuracy)

        else:
            print("Loading saved data...")
            classifier = load(generate_model_path(dataset_name, MODEL_TYPE))
            x_train = load(generate_x_train_path(dataset_name, MODEL_TYPE))
            y_train = load(generate_y_train_path(dataset_name, MODEL_TYPE))
            x_test = load(generate_x_test_path(dataset_name, MODEL_TYPE))
            y_test = load(generate_y_test_path(dataset_name, MODEL_TYPE))
            print("Loaded saved data")
            num_classes = len(set(y_test))

        num_features = len(x_test[0])
        _write_num_features(num_features)

        if MODEL_TYPE == 'RF':
            forest = RandomForest.from_sklearn(classifier, num_classes)
        else:
            initial_prediction = predict_initial(y_train, num_classes)
            forest = GradientBoostingForest.from_sklearn(classifier, num_classes, learning_rate, initial_prediction)

    x_test, y_test = take_n(classifier, x_test, y_test, get_num_adv())
    load_or_train_time = round(time.time() - start_0, TIME_PRECISION)

    ae_lookup = None
    lookup_creation_time = 0
    if CREATE_LOOKUP:
        start = time.time()
        ae_lookup = create_ae_lookup(x_train, y_train, num_features, num_classes)
        lookup_creation_time = round(time.time()-start, TIME_PRECISION)

    if constants.PRINT_NUMBER_OF_LEAVES:
        nodes = [e for e in forest.walk()]
        num_leaves = 0
        for e in nodes:
            if e.child_left.is_leaf:
                num_leaves += 1
            if e.child_right.is_leaf:
                num_leaves += 1
        print('Number of leaves: ', num_leaves)

    print(f'Start generating {OUTPUT_FILE} ...')
    start = time.time()
    forest.generate_cc(OUTPUT_FILE, x_test, y_test, epsilon, num_mutations, dataset_name=dataset_name)
    file_generation_time = round(time.time() - start, TIME_PRECISION)
    print(f'Finished generating {OUTPUT_FILE}')

    write_num_checks(dataset_name, len(y_test))

    if constants.SKIP_COMPILATION:
        compilation_time = 0.0
    else:
        compilation_time = compile_fuzz_target()

    if constants.COMPILE_ONLY:
        return

    t = get_execution_time(dataset_name, MODEL_TYPE)
    start_investigate = time.time()

    if INVESTIGATE:
        runs = []
        if INVESTIGATE_WITH_SCATTER:
            n = 20
            # steps = [500, 1000, 5000, 10000, 20000, 50000, 100000]
            steps = [100000, 200000, 300000]
            for step in steps:
                runs += n*[step]
        else:
            # runs = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 400000] * NUM_INVESTIGATE_RUNS
            runs = [500, 1000, 2500, 5000, 10000, 12500, 15000, 20000, 25000, 30000, 40000] * NUM_INVESTIGATE_RUNS
            # runs = [*[i * 1000 for i in range(1, 21)], *[20000 + i * 5000 for i in range(1, 37)],
            #         *[200000 + i * 25000 for i in range(1, 9)]] * NUM_INVESTIGATE_RUNS
        # runs = [1000, 2000]
    else:
        runs = [-1]  # Unlimited

    results_per_n_runs = []
    fuzzing_time = 0
    avg_time_fuzzing = 0
    combined_avg_time_fuzzing = 0
    process_ae_time = 0
    avg_execs = 0
    avg_coverage = 0
    for num_runs in runs:
        start = time.time()
        make_or_empty_dir(ADV_DIR)

        if FUZZ_ONE_POINT_PER_INSTANCE:
            time_fuzzing, avg_time_fuzzing, avg_coverage, avg_execs = run_single_datapoints(x_test, y_test, t,
                                                                                            ae_lookup, num_runs,
                                                                                            num_classes)
        else:
            time_fuzzing, avg_time_fuzzing, avg_coverage = run_multi_datapoints(x_test, y_test, t, ae_lookup,
                                                                                num_runs, num_classes)

        print(f'Actual running time was {time_fuzzing} seconds')
        print(f'Average fuzzing time is {avg_time_fuzzing}')
        print("Finished fuzzing")
        print()

        fuzzing_time = round(time.time() - start, TIME_PRECISION)
        combined_avg_time_fuzzing = round(fuzzing_time/len(y_test), TIME_PRECISION)

        from result import process_adversarial_examples
        process_ae_time, avg_best_norm, num_attacked, avg_dist_to_own_class, avg_dist_to_other_class = \
            process_adversarial_examples(dataset_name, MODEL_TYPE, epsilon, num_features, ITERATION, len(y_test),
                                         ae_lookup)
        if INVESTIGATE:
            perc_attacked = num_attacked/len(y_test)
            avg_coverage /= 100
            info = [num_runs, avg_best_norm, perc_attacked, avg_coverage, combined_avg_time_fuzzing,
                    avg_dist_to_own_class, avg_dist_to_other_class]
            results_per_n_runs.append(info)
            print(*info)
        else:
            print('Avg dist to own class: ', avg_dist_to_own_class)
            print('Avg dist to other class: ', avg_dist_to_other_class)

    if INVESTIGATE:
        coverage_str = ""
        for results_per_run in results_per_n_runs:
            coverage_str += ",".join([str(e) for e in results_per_run]) + "\n"
        write_file(generate_coverages_path(dataset_name, epsilon, MODEL_TYPE, INVESTIGATE_WITH_SCATTER), coverage_str)

        print(f'The complete investigate execution took {time.time()-start_investigate} seconds')
        if INVESTIGATE_WITH_SCATTER:
            plot_scatter(dataset_name, epsilon, MODEL_TYPE)
        else:
            plot_lines(dataset_name, epsilon, MODEL_TYPE)
    else:
        total_time = time.time()-start_0
        other_time = round(total_time - compilation_time - file_generation_time - lookup_creation_time - fuzzing_time
                           - process_ae_time - load_or_train_time + 0.0009, TIME_PRECISION)  # Make > 0.0 (rounding)
        total_time = round(total_time, TIME_PRECISION)

        write_execution_details(dataset_name, len(y_test), avg_time_fuzzing, combined_avg_time_fuzzing,
                                load_or_train_time, lookup_creation_time, file_generation_time, compilation_time,
                                fuzzing_time, other_time, total_time, avg_coverage, avg_execs)

        if len(runs) == 1:
            # Normal execution
            print()
            print('Real avg fuzzing time    : ', avg_time_fuzzing)
            print('Combined avg fuzzing time: ', combined_avg_time_fuzzing)
            print()
            print('Loading and training time: ', load_or_train_time)
            print('ANN lookup creation time : ', lookup_creation_time)
            print('File generation time     : ', file_generation_time)
            print('Compilation time         : ', compilation_time)
            print('Fuzzing time             : ', fuzzing_time)
            print('Process AE time          : ', process_ae_time)
            print('Other time               : ', other_time)
            print('Total time               : ', total_time)
        if constants.MEASURE_EXEC_P_S:
            print('Avg Executions per second: ', avg_execs)


def compile_fuzz_target():
    start = time.time()

    if constants.FUZZER == 'libFuzzer':
        _compile_libfuzzer_target()
    elif constants.FUZZER == 'AFL++':
        _compile_afl_pp_mutation()
        _compile_afl_pp_target()
    elif constants.FUZZER == 'AFLGo':
        make_or_empty_dir("temp")
        _create_afl_go_directed_targets()
        _afl_go_generate_cg()
        _compile_afl_go_target()
    elif constants.FUZZER == 'honggfuzz':
        _compile_honggfuzz_target()
    else:
        raise NotImplementedError('Unknown fuzzer when compiling')

    end = time.time()
    return round(end-start, TIME_PRECISION)


def _compile_honggfuzz_target():
    command = [constants.HONG_COMPILER_PATH, OUTPUT_FILE,
               # '-fsanitize-coverage=trace-pc-guard,indirect-calls,trace-cmp',  # Auto inserted by honggfuzz
               '-o', 'hongg_fuzzme']
    subprocess.run(command, check=True)  # Raises error if command exitcode != 0


def _compile_libfuzzer_target():
    file_size = os.path.getsize(OUTPUT_FILE)
    if not constants.NEVER_OPTIMIZE and (constants.ALWAYS_OPTIMIZE or file_size / 1024 < 1024):  # < 1 MB
        print('Compiling with optimization')
        optimize_option = "-O3"
    else:
        print('Compiling without optimization')
        optimize_option = "-O0"

    command = ['clang++', optimize_option, '-fbracket-depth=1100']

    if not constants.TEST_OUTSIDE_FUZZER:
        command += ['-fsanitize=fuzzer', '-fsanitize-coverage=bb']

        if constants.USE_CUSTOM_MUTATOR:
            command.append('-fsanitize-coverage=bb')
        else:
            command.append('-fsanitize-coverage=bb,trace-cmp')

        if constants.MEASURE_COVERAGE or constants.INVESTIGATE:
            command += ['-fprofile-instr-generate', '-fcoverage-mapping']
        # command = ["./compile_target.sh", OUTPUT_FILE, optimize_option]

        if FUZZ_WITHOUT_COVERAGE_GUIDANCE:
            allowlist_name = "allowlist_none.txt"
        elif constants.FUZZ_ONLY_COV_FOR_FOREST:
            allowlist_name = "allowlist_trees.txt"
        elif constants.FUZZ_ONLY_COV_FOR_CHECK:
            allowlist_name = "allowlist_check.txt"
        else:
            allowlist_name = None

        if allowlist_name is not None:
            command += [f'-fsanitize-coverage-allowlist={allowlist_name}', '-fsanitize-coverage-blocklist=blocklist.txt']
    else:
        command.append('-std=c++17')
    command += [OUTPUT_FILE, '-o', 'fuzzme']

    # env = {
    #     "LLVM_ENABLE_THREADS": "1"
    # }

    subprocess.run(command, check=True)  # Raises error if command exitcode != 0


def _compile_afl_pp_mutation():
    command = ["./afl_compile_mutation.sh", constants.AFL_MUTATE_FILENAME]
    subprocess.run(command, check=True)  # Raises error if command exitcode != 0


def _compile_afl_pp_target():
    remove_file(constants.AFLPP_DICT_PATH)
    environ = {
        "AFL_DONT_OPTIMIZE": f'{int(not constants.ALWAYS_OPTIMIZE)}',
        "AFL_LLVM_DICT2FILE": constants.AFLPP_DICT_PATH,
        # https://github.com/AFLplusplus/AFLplusplus/blob/stable/instrumentation/README.laf-intel.md
        "AFL_LLVM_LAF_ALL": "1",
    }
    command = [constants.AFLPP_COMPILER_PATH, OUTPUT_FILE, '-o', 'afl_fuzzme']
    proc = subprocess.Popen(" ".join(command), shell=True, env=environ)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError('Compilation failed')

    if constants.AFL_USE_CMP_LOG:
        environ.update({"AFL_LLVM_CMPLOG": "1"})
        environ.pop("AFL_LLVM_DICT2FILE", None)  # Make sure dict file is not generated twice
        command = [constants.AFLPP_COMPILER_PATH, OUTPUT_FILE, '-o', 'afl_fuzzme.cmplog']
        proc = subprocess.Popen(" ".join(command), shell=True, env=environ)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError('Compilation failed')


def _compile_afl_go_target():
    command = [constants.AFL_GO_COMPILER_PATH, OUTPUT_FILE, '-distance=distance.cfg.txt', '-o', 'afl_fuzzme']
    subprocess.run(command, check=True)  # Raises error if command exitcode != 0


def _afl_go_generate_cg():
    command = [constants.AFL_GO_COMPILER_PATH, f'-targets={constants.AFL_GO_TARGETS_FILE}',
               f'-outdir={join(os.getcwd(), "temp")}', '-flto', '-fuse-ld=gold',
               '-Wl,-plugin-opt=save-temps', '-shared', '-fPIC',
               '-lpthread', "-v", OUTPUT_FILE, '-o', 'afl_fuzzme']
    subprocess.run(command, check=True)
    # FUTURE add cleanup
    gen_dist_command = [constants.AFL_GO_GEN_DIST_PATH, os.getcwd(),
                        join(os.getcwd(), 'temp'), 'afl_fuzzme']
    subprocess.run(gen_dist_command, check=True)


def _create_afl_go_directed_targets():
    last_step = constants.DISTANCE_STEPS[-1]
    search_for = f'{last_step}.txt'
    command = ['./find_target_lines.sh', search_for, OUTPUT_FILE]

    comprocc = subprocess.run(command, stdout=subprocess.PIPE)
    out_num = comprocc.stdout.decode('utf-8')
    out_num = out_num.strip()  # remove trailing \n
    out_str = f"{OUTPUT_FILE}:{out_num}"
    write_file(join("temp", constants.AFL_GO_TARGETS_FILE), out_str)


def create_ae_lookup(x_train, y_train, num_features, num_classes):
    """
    returns {class_id: {"instances": target_instances, "ann": ann_tree}}
        where ann_tree is build with target_instances: all instances with label != class_id
    """
    lookup_per_class = dict()

    # Possible distances:
    # euclidean
    # manhattan: The distance between two points measured along axes at right angles
    # cosine: equals the cosine of the angle between two vectors,
    #   - equivalent to the inner product of the same vectors normalized to both have length 1
    #   - equivalent to the euclidean distance of normalized vectors = sqrt(2-2*cos(u,v))
    # hamming distance: the number of positions at which the symbols are different
    # dot (inner) product distance - the sum of the products of the corresponding entries
    #   - the product of the Euclidean magnitudes of the two vectors and the cosine of the angle between them
    #   - https://en.wikipedia.org/wiki/Dot_product

    if num_features == 784:
        metric = 'hamming'
    else:
        metric = 'euclidean'

    if len(y_train) > MAX_POINTS_LOOKUP:
        indices = _n_indices(len(y_train), MAX_POINTS_LOOKUP)
        x_train = [x_train[i] for i in indices]
        y_train = [y_train[i] for i in indices]

    start = time.time()
    for i in range(num_classes):
        target_instances = [fs for fs, y in zip(x_train, y_train) if y != i]
        ann_tree = AnnoyIndex(num_features, metric)
        for target_id, e in enumerate(target_instances):
            ann_tree.add_item(target_id, e)
        ann_tree.build(ANN_TREES)
        lookup_per_class.update({i: {"instances": target_instances, "ann": ann_tree}})
    end = time.time()
    print("Building ANN lookup took ", round(end-start, 3), " seconds")
    return lookup_per_class


def run_single_datapoints(x_test, y_test, t, ae_lookup, num_runs, num_classes):
    num_features = len(x_test[0])
    q = Queue()
    avg_best_l_inf = 0
    for check_num, (original_features, original_class) in enumerate(zip(x_test, y_test)):
        original_features = np.array(original_features)
        corpus_points = None
        if INITIALIZE_WITH_FULL_TRAIN_SET:
            corpus_points = ae_lookup[original_class]["instances"]
        elif constants.INITIALIZE_WITH_AVG_OPPOSITE and num_classes == 2:
            all_points_other_class = ae_lookup[original_class]["instances"]
            avg = np.zeros(len(all_points_other_class[0]))
            for p in all_points_other_class:
                avg += p
            corpus_points = [avg / len(all_points_other_class)]
        elif INITIALIZE_WITH_AE:
            instances = ae_lookup[original_class]["instances"]
            closest_ae_indices = ae_lookup[original_class]["ann"].get_nns_by_vector(original_features, K_ANN)
            corpus_points = [instances[closest_ae_index] for closest_ae_index in closest_ae_indices]
            avg_best_l_inf += np.min(np.max(np.abs(np.array(corpus_points) - np.array(original_features))))
            if constants.INITIALIZE_WITH_POINT_IN_BETWEEN:
                for ae in corpus_points.copy():
                    vic = np.array(original_features)
                    diff_vec = np.array(ae) - vic
                    corpus_points.append(vic+diff_vec/2)
                    if constants.INITIALIZE_WITH_EXTRA_POINTS_IN_BETWEEN:
                        corpus_points.append(vic+diff_vec/4)
                        corpus_points.append(vic+(3*diff_vec/4))

        q.put((check_num, original_features, original_class, corpus_points))

    if INITIALIZE_WITH_AE:
        print("avg AE dist: ", avg_best_l_inf / len(y_test))

    if constants.FUZZER == 'AFL++':
        make_or_empty_dir(constants.AFL_OUTPUT_DIR)  # Because AFL may otherwise think the dir is in use

    if constants.FUZZER == 'honggfuzz':
        make_or_empty_dir(constants.HONG_OUTPUT_DIR)

    print()
    print(f"Fuzzing {len(y_test)} points")
    print("Started fuzzing...")
    print(f'Expected running time is {q.qsize()*(t+1)/NUM_THREADS} seconds')

    manager = Manager()
    processes = []
    coverages = manager.list()
    execs = manager.list()
    start = time.time()
    for i in range(NUM_THREADS):
        p = Process(target=runner, args=(i, q, num_features, t, num_runs, coverages, execs,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    coverages = list(coverages)
    execs = list(execs)
    avg_execs = statistics.mean(execs) if len(execs) > 0 else 0.0
    avg_coverage = statistics.mean(coverages) if len(coverages) > 0 else 0.0

    if DOUBLE_FUZZ_WITH_AE:
        from result import find_ids_no_ae
        ids_no_ae = find_ids_no_ae(len(y_test))
        print(f"Now fuzzing {len(ids_no_ae)} victims again with an AE as init")
        for point_id in ids_no_ae:
            original_features = x_test[point_id]
            original_class = y_test[point_id]
            instances = ae_lookup[original_class]["instances"]
            closest_ae_indices = ae_lookup[original_class]["ann"].get_nns_by_vector(original_features, K_ANN)
            corpus_points = [instances[closest_ae_index] for closest_ae_index in closest_ae_indices]
            # if constants.INITIALIZE_WITH_POINT_IN_BETWEEN:
            #     corpus_points += [(np.array(e)+original_features) / 2 for e in corpus_points]
            if constants.INITIALIZE_WITH_POINT_IN_BETWEEN:
                for ae in corpus_points.copy():
                    vic = np.array(original_features)
                    diff_vec = np.array(ae) - vic
                    corpus_points.append(vic+diff_vec/2)
                    if constants.INITIALIZE_WITH_EXTRA_POINTS_IN_BETWEEN:
                        corpus_points.append(vic+diff_vec/4)
                        corpus_points.append(vic+(3*diff_vec/4))
            q.put((point_id, original_features, original_class, corpus_points))

        if constants.FUZZER == 'AFL++':
            make_or_empty_dir(constants.AFL_OUTPUT_DIR)  # Because AFL may otherwise think the dir is in use

        if constants.FUZZER == 'honggfuzz':
            make_or_empty_dir(constants.HONG_OUTPUT_DIR)

        for i in range(NUM_THREADS):
            p = Process(target=runner, args=(i, q, num_features, t, num_runs, coverages, execs,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    end = time.time()
    time_fuzzing = end-start
    avg_time_fuzzing = round(time_fuzzing/len(y_test), TIME_PRECISION)
    time_fuzzing = round(time_fuzzing, TIME_PRECISION)
    return time_fuzzing, avg_time_fuzzing, avg_coverage, avg_execs


def runner(runner_id, q: Queue, num_features, t, num_runs, coverages, execs):
    if not constants.USE_CUSTOM_MUTATOR:
        fuzzer_commands = ['-use_value_profile=1',
                           # '-data_flow_trace=1',
                           '-use_cmp=1',
                           f'-max_total_time={t}',
                           ]
    else:
        fuzzer_commands = [f"-max_total_time={t}",
                           "-use_cmp=0"]
    ent = "0" if constants.NO_ENTROPIC else "1" if constants.FORCE_ENTROPIC else get_entropic(_dataset)
    fuzzer_commands += [f"-mutate_depth={MUTATE_DEPTH}",
                        "-reduce_inputs=0",
                        "-prefer_small=0",
                        f"-max_len={num_features * BYTES_PER_FEATURE}",
                        f"-runs={num_runs}",
                        "-reload=0",
                        f'-entropic={ent}',
                        "-len_control=0",
                        "-print_coverage=0",
                        "-print_pcs=0",
                        # "-help=1",
                        ]
    while True:
        try:
            (check_num, original_features, original_class, corpus_points) = q.get(timeout=0.1)
        except Empty:
            # print(f'Runner {runner_id} is quitting')
            return

        queue_size = q.qsize()

        if not INVESTIGATE:
            # Normal execution
            print(queue_size, " points left")

        corpus_dir = CORPUS_DIR+str(runner_id)
        corpus_features = [original_features, *corpus_points] if corpus_points is not None else [original_features]
        if constants.NO_SEED_INIT:
            generate_corpus([[0.0]*num_features], corpus_dir)
        else:
            generate_corpus(corpus_features, corpus_dir)

        if constants.FUZZER in ['libFuzzer', 'honggfuzz']:
            fuzzer_initialize_commands = [str(check_num), str(num_features), str(original_class)]
            for ft in original_features:
                fuzzer_initialize_commands.append(str(ft))

            command = ["./fuzzme"]
            command += fuzzer_initialize_commands
            if constants.FUZZER == 'libFuzzer':
                if not constants.TEST_OUTSIDE_FUZZER:
                    command.append(corpus_dir)
                    command += fuzzer_commands
                if constants.TEST_OUTSIDE_FUZZER and (INITIALIZE_WITH_AE or DOUBLE_FUZZ_WITH_AE):
                    command.append(corpus_dir)
                coverage, exec_p_s = _run_lib_fuzzer(command, runner_id)
            else:
                coverage, exec_p_s = _run_honggfuzz(fuzzer_initialize_commands, tt, num_features, runner_id=runner_id)
        elif constants.FUZZER in ['AFL++', 'AFLGo']:
            coverage, exec_p_s = _run_afl(check_num, num_features, original_class, original_features, tt, runner_id,
                                          fid=queue_size)
        else:
            coverage = 0
            exec_p_s = 0
            ...

        coverages.append(float(coverage))
        execs.append(exec_p_s)


def run_multi_datapoints(x_test, y_test, t, ae_lookup, num_runs, num_classes):
    if constants.FUZZER != 'libFuzzer':
        raise ValueError('Running with multiple data-points only possible for libFuzzer')

    avg_best_l_inf = 0
    corpus_points = []
    if INITIALIZE_WITH_FULL_TRAIN_SET:
        for class_id in range(num_classes):
            corpus_points += ae_lookup[class_id]["instances"]
    elif INITIALIZE_WITH_AE:
        for check_num, (original_features, original_class) in enumerate(zip(x_test, y_test)):
            instances = ae_lookup[original_class]["instances"]
            closest_ae_indices = ae_lookup[original_class]["ann"].get_nns_by_vector(original_features, K_ANN)
            corpus_points += [instances[closest_ae_index] for closest_ae_index in closest_ae_indices]
            avg_best_l_inf += np.min(np.max(np.abs(np.array(corpus_points) - np.array(original_features))))
        print("avg AE dist: ", avg_best_l_inf / len(y_test))

    num_points = len(y_test)
    running_time = ceil(t*num_points/NUM_THREADS)
    print()
    print(f"Fuzzing {num_points} points")
    print("Started fuzzing...")
    print(f'Expected running time is {running_time} seconds')

    num_features = len(x_test[0])
    fuzzer_commands = [f"-mutate_depth={MUTATE_DEPTH}",
                       "-len_control=0",
                       "-reduce_inputs=0",
                       f"-max_len={num_features * BYTES_PER_FEATURE}",
                       f"-max_total_time={running_time}",
                       f"-jobs={NUM_THREADS}",
                       f"-workers={NUM_THREADS}",
                       f"-runs={num_runs}",
                       "-reload=1",
                       "-print_coverage=0",
                       "-print_pcs=0"]
    command = ["./fuzzme"]

    start = time.time()
    corpus_features = [*x_test, *corpus_points]

    if constants.NO_SEED_INIT:
        generate_corpus([[0.0]*num_features], CORPUS_DIR)
    else:
        generate_corpus(corpus_features, CORPUS_DIR)

    command.append(CORPUS_DIR)
    command += fuzzer_commands

    coverage, _ = _run_lib_fuzzer(command, runner_id=None)
    end = time.time()

    time_fuzzing = end-start
    avg_time_fuzzing = round(time_fuzzing/len(y_test), TIME_PRECISION)
    time_fuzzing = round(time_fuzzing, TIME_PRECISION)
    return time_fuzzing, avg_time_fuzzing, coverage


def _run_lib_fuzzer(command, runner_id=None):
    runner_id_str = "" if runner_id is None else str(runner_id)

    profraw_name = f'.fuzzme{runner_id_str}.profraw'
    profdata_name = f'.fuzzme{runner_id_str}.profdata'
    environment = dict(os.environ, LLVM_PROFILE_FILE=profraw_name)

    exec_p_s = 0.0
    if constants.MEASURE_EXEC_P_S:
        if constants.TEST_OUTSIDE_FUZZER:
            comprocc = subprocess.run(command, stdout=subprocess.PIPE, env=environment)
            run_output_lines = comprocc.stdout.decode('utf-8').splitlines()
        else:
            comprocc = subprocess.run(command, stderr=subprocess.PIPE, env=environment)
            run_output_lines = comprocc.stderr.decode('utf-8').splitlines()
        try:
            last_line_splitted = run_output_lines[-1].split(' ')
            num_exec = int(last_line_splitted[1])
            num_sec = float(last_line_splitted[4])
            exec_p_s = num_exec/num_sec
        except (IndexError, ValueError, ZeroDivisionError):
            exec_p_s = 0.0
    else:
        if SHOW_OUTPUT:
            subprocess.run(command, env=environment)
        else:
            # LibFuzzer also writes output to stderr unfortunately
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment)

    coverage = 0
    if INVESTIGATE or constants.MEASURE_COVERAGE:
        subprocess.run(['llvm-profdata', 'merge', '-sparse', profraw_name, '-o', profdata_name])
        coverage_output_lines = subprocess.run(
            ['llvm-cov', 'report', 'fuzzme', f'-instr-profile={profdata_name}'],
            stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()

        info = coverage_output_lines[-1].split()
        coverage = info[3].replace("%", "")  # -1 is line coverage, 3 is region coverage
    return float(coverage), exec_p_s


def _run_honggfuzz(initialize_commands, t, num_features, runner_id=None, fid=None):
    runner_id_str = "" if runner_id is None else str(runner_id)
    corpus_dir = CORPUS_DIR + runner_id_str
    if fid is not None:
        fuzzer_name = "main" + str(fid)  # Do not use same fuzzer_name twice in same run_fate run
    else:
        fuzzer_name = "main" + runner_id_str
    fuzzer_dir = join(constants.HONG_OUTPUT_DIR, fuzzer_name)

    command = [constants.HONG_FUZZER_PATH, '-i', corpus_dir, '--output', fuzzer_dir, '-P',
               '-n', '1', '--mutations_per_run', str(constants.MUTATE_DEPTH),
               '-F', str(constants.BYTES_PER_FEATURE * num_features)]
    if constants.FUZZ_WITHOUT_COVERAGE_GUIDANCE:
        command.append('--noinst')
    else:
        command.append('--instrument')
    if constants.DEBUG:
        command += ['-v', '-d']
    else:
        command.append('-q')

    command += ['--run_time', str(t)]
    command += ['--', './hongg_fuzzme', *initialize_commands]

    exec_p_s = 0.0
    coverage = 0
    if SHOW_OUTPUT:
        subprocess.run(command)
    else:
        # LibFuzzer also writes output to stderr unfortunately
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return float(coverage), exec_p_s


def _run_afl(check_num, num_features, original_class, original_features, t, runner_id=None, fid=None):
    t = int(t)
    runner_id_str = "" if runner_id is None else str(runner_id)
    corpus_dir = CORPUS_DIR + runner_id_str
    if fid is not None:
        fuzzer_name = "main" + str(fid)  # Do not use same fuzzer_name twice in same run_fate run
    else:
        fuzzer_name = "main" + runner_id_str
    # -l 2 sets cmplog options 2 means larger files
    command = ['afl-fuzz', '-l', '2', '-i', corpus_dir, '-o', constants.AFL_OUTPUT_DIR]
    if constants.AFL_SCHEDULE is not None:
        command += ['-p', constants.AFL_SCHEDULE]
    if constants.SKIP_DETERMINISTIC:
        # -d skip deterministic phase
        command.append('-d')
    if constants.ENABLE_DETERMINISTIC:
        # -D enable deterministic fuzzing (once per queue entry)
        command.append('-D')

    command += ['-V', str(t)]

    if constants.AFL_USE_CMP_LOG and not constants.USE_CUSTOM_MUTATOR:
        command += ['-c', './afl_fuzzme.cmplog', '-m', 'none']
    if constants.AFL_USE_DICT and not constants.USE_CUSTOM_MUTATOR:
        command += ['-x', './afl_dict']
    if constants.FUZZ_WITHOUT_COVERAGE_GUIDANCE:
        command.append('-n')
    else:
        command += ['-M', fuzzer_name]
    command += ['--', './afl_fuzzme']

    environment = dict(os.environ)
    afl_env = {
        "AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES": "1",
        "AFL_SKIP_CPUFREQ": "1",
        "AFL_EXIT_ON_TIME": "6",
        "AFL_NO_AFFINITY": "1",
        # "AFL_DEBUG": f"{int(constants.DEBUG)}",
        "AFL_DEBUG_CHILD": f"{int(constants.DEBUG)}",
        "AFL_TESTCACHE_SIZE": "50",
        "CHECK_NUM": str(check_num),
        "NUM_FEATURES": str(num_features),
        "ORIGINAL_CLASS": str(original_class),
        "ORIGINAL_FEATURES": " ".join([str(e) for e in original_features])
    }
    environment.update(afl_env)

    if constants.LIMIT_TIME:
        environment.update({"AFL_FAST_CAL": "1"})  # limit the calibration stage to three cycles for speedup

    if constants.USE_CUSTOM_MUTATOR:
        mutator_env = {
            "AFL_DISABLE_TRIM": "1",
            "AFL_NO_AUTODICT": "1",
            "AFL_CUSTOM_MUTATOR_ONLY": "1",
            "AFL_CUSTOM_MUTATOR_LIBRARY": "afl_mutate.so",
        }
        environment.update(mutator_env)
    if not constants.DEBUG:
        no_debug_env = {
            "AFL_NO_UI": "1",
            "AFL_QUIET": "1"
        }
        environment.update(no_debug_env)

    exec_p_s = 0.0
    coverage = 0.0
    if SHOW_OUTPUT:
        subprocess.run(command, env=environment)
    else:
        # AFL++ also writes output to stderr unfortunately
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=environment)

    if constants.MEASURE_EXEC_P_S or INVESTIGATE:
        try:
            filepath = join(constants.AFL_OUTPUT_DIR, fuzzer_name, 'fuzzer_stats')
            with open(filepath, 'r') as file:
                contents = file.read()
            lines = contents.splitlines()
            for line in lines:
                parts = line.split(":")
                fuzzer_name = parts[0].strip()
                if fuzzer_name == 'execs_done':
                    val = parts[1].strip()
                    exec_p_s = round(int(val) / t, 2)
                elif fuzzer_name == 'bitmap_cvg':
                    val = parts[1].strip()[:-1]
                    coverage = float(val)
        except:
            return coverage, exec_p_s

    return coverage, exec_p_s


def predict_initial(y_train, num_classes: int):
    prior = Counter(y_train)
    if num_classes == 2:
        prior_count_0 = prior[0]
        prior_count_1 = prior[1]
        return np.array([np.log(prior_count_1 / prior_count_0)])
    else:
        return np.log([prior[i] / len(y_train) for i in range(num_classes)])


def generate_corpus(x, corpus_dir):
    make_or_empty_dir(corpus_dir)

    for i, fs in enumerate(x):
        filename = str(i + 1)
        b = struct.pack('d' * len(fs), *fs)  # use 'f' for float and 'd' for double

        with open(os.path.join(corpus_dir, filename), 'wb+') as file:
            file.write(b)

        # For verification purposes
        # fs = ["%.6f" % f for f in fs]
        # with open("checkfile", "w+") as f2:
        #     f2.write(",".join(fs))
        #     float_array = array('f', fs)
        #     float_array.tofile(file)


class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawTextHelpFormatter):
    pass


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description="Fuzzing for Adversarial Examples in Tree Ensemble models",
                                      formatter_class=RawTextArgumentDefaultsHelpFormatter)
    _parser.add_argument('dataset', type=str, choices=DATASETS,
                         help="the identifier of the dataset to fuzz.\noptions: " + ", ".join(DATASETS),
                         metavar="dataset")
    _parser.add_argument('model_type', type=str, choices=MODEL_TYPES,
                         help="the type of the model.\noptions: " + ", ".join(MODEL_TYPES),
                         metavar="model_type")
    _parser.add_argument('epsilon', type=float, nargs='?', default=None,
                         help="epsilon range for mutation. If not supplied uses default or best epsilon best dataset")
    _parser.add_argument('i', type=int, nargs='?', default=0, help="to differentiate between multiple repetitions")
    _parser.add_argument('-q', '--quick', dest='quick', action='store_true', default=False,
                         help=f"Execute for {constants.NUM_ADV_QUICK} instead of {NUM_ADV_CHECKS} victims")
    _parser.add_argument('-qq', '--super_quick', dest='super_quick', action='store_true', default=False,
                         help=f"Execute for {constants.NUM_ADV_SUPER_QUICK} instead of {NUM_ADV_CHECKS} victims")
    _parser.add_argument('--reload', dest='reload', action='store_true', default=False,
                         help="should be included on first run for new data or training settings")
    _parser.add_argument('-po', '--parse_only', dest='parse_only', action='store_true', default=False,
                         help="only parse previous results")
    _parser.add_argument('-nc', '--no_compile', dest='no_compile', action='store_true', default=False,
                         help="do not compile target")
    _args = _parser.parse_args()

    _dataset = _args.dataset
    MODEL_TYPE = _args.model_type
    _eps = _args.epsilon
    if _eps is None:
        # only overwrite epsilon if not provided by the user
        _eps = get_epsilon(_dataset)

    ITERATION = _args.i
    RELOAD = _args.reload

    if _args.parse_only:
        from result import process_adversarial_examples
        process_adversarial_examples(_dataset, MODEL_TYPE, _eps, read_num_features(), ITERATION, num_points=None)
        exit(0)

    if _args.super_quick:
        constants.NUM_ADV_CHECKS = constants.NUM_ADV_SUPER_QUICK
    elif _args.quick:
        constants.NUM_ADV_CHECKS = constants.NUM_ADV_QUICK

    if _args.no_compile:
        constants.SKIP_COMPILATION = True

    if _dataset == 'GROOT':
        generate_model_and_run(_dataset, _eps, from_json=True,
                               json_filename=join(MODEL_DIR, 'mnist_groot_rf.json'))
    else:
        generate_model_and_run(_dataset, _eps)
