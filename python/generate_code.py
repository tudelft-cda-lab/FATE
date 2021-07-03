import glob
from os.path import join

from constants import CHECK_DIR, DISTANCE_NORM, DISTANCE_NORMS, DISTANCE_STEPS, \
    FUZZ_ONE_POINT_PER_INSTANCE, COMBINE_DISTANCE_AND_PROBABILITY, PROBABILITY_STEPS, USE_PROBABILITY_STEPS, \
    WRITE_AE_ALWAYS_IN_IF, WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES, USE_PROBABILITY_STEPS_SPECIAL, \
    PROBA_SPECIAL_START_STEP, \
    PROBA_SPECIAL_STEP_SIZE, PROBA_SPECIAL_ALWAYS, PROBA_LIMIT_WITHIN_EPSILON, TEST_OUTSIDE_FUZZER
from utils import make_or_empty_dir


def _generate_estimator_calls(num_trees):
    res = f"\tdouble** tree_outputs = 0;\n"
    res += f"\ttree_outputs = new double*[{num_trees}];\n"
    for i in range(num_trees):
        res += f"\ttree_outputs[{i}] = tree_{i}(fuzzed_features);\n"

    res += f"\tauto probs = calc_class_prob(tree_outputs, {num_trees});\n"
    res += "delete[] tree_outputs;\n"
    return res


def generate_predict_probs(num_trees):
    res = "double* predict_probs(double fuzzed_features[]) {\n"
    res += _generate_estimator_calls(num_trees)
    res += "return probs;\n}\n"
    return res


def _generate_adversarial_checks(x_test, y_test, num_classes, epsilon):
    make_or_empty_dir(CHECK_DIR)
    res = ""
    if FUZZ_ONE_POINT_PER_INSTANCE:
        for check_num, original_features in enumerate(x_test):
            _write_check_file(check_num, original_features)
        res += __generate_generic_adversarial_check(len(x_test[0]), num_classes, epsilon)
    else:
        for check_num, original_features in enumerate(x_test):
            _write_check_file(check_num, original_features)
            expected_outcome = y_test[check_num]
            res += __generate_adversarial_check_for_datapoint(original_features, expected_outcome, check_num)
    return res


def __generate_generic_adversarial_check(num_features, num_classes, epsilon):
    res = f"void check_for_ae(double fuzzed_features[], int fuzzed_class, double probs[]){{\n"
    res += f"int num_features = {num_features};\n"
    res += f"double proba_original = probs[original_class];\n"
    res += f'std::sort(probs, probs+{num_classes}, std::greater<double>{{}});\n'
    res += 'std::string check_base = "check_";\n'
    res += f'std::string file_base = check_base.append(to_string(check_num)).append("-");\n'
    if DISTANCE_NORM not in DISTANCE_NORMS:
        raise ValueError(f"Unknown distance function '{DISTANCE_NORM}'")
    if DISTANCE_NORM == 'l_inf':
        res += "double distance = distance_linf(fuzzed_features);\n"
    else:
        raise NotImplementedError("Distance norm not implemented")
    if TEST_OUTSIDE_FUZZER:
        res += 'if (fuzzed_class != original_class){\ncur_ae_dist = distance;\n} else {\ncur_ae_dist = 1.1;\n}'
    if COMBINE_DISTANCE_AND_PROBABILITY:
        res += "distance = distance + proba_original;\n"
    res += f"if (fuzzed_class != original_class) {{\n"
    res += __generate_step_string(DISTANCE_STEPS)
    if USE_PROBABILITY_STEPS:
        res += "\n} else { \n"
        res += __generate_proba_steps(PROBABILITY_STEPS)
        res += "\n}"
    elif USE_PROBABILITY_STEPS_SPECIAL:
        if PROBA_SPECIAL_ALWAYS:
            res += "\n}" + __generate_proba_steps_special(epsilon)
        else:
            res += "\n} else { \n"
            res += __generate_proba_steps_special(epsilon)
            res += "\n}"
    else:
        res += "\n}"
    res += "\n}\n"
    return res


def __generate_adversarial_check_for_datapoint(original_features, original_class, num):
    res = f"void check_{num}(double fuzzed_features[], int fuzzed_class, double probs[]){{\n"
    res += f"int original_class = {original_class};"
    res += f"double proba_original = probs[original_class];\n"
    res += f"int num_features = {len(original_features)};\n"
    res += f"double original_stack[] = {_array_initialize(original_features)};\n"
    res += f"double* original_features = new double[{len(original_features)}];\n"
    res += "copy(original_stack, original_stack+num_features, &original_features[0]);\n"
    if DISTANCE_NORM not in DISTANCE_NORMS:
        raise ValueError(f"Unknown distance function '{DISTANCE_NORM}'")
    if DISTANCE_NORM == 'l_inf':
        res += "double distance = distance_linf(fuzzed_features, original_features);\n"
    else:
        raise NotImplementedError("Distance norm not implemented")
    if COMBINE_DISTANCE_AND_PROBABILITY:
        res += "distance = distance + proba_original;\n"
    res += "delete[] original_features;\n"
    res += f"if (fuzzed_class != {original_class}) {{\n"
    res += __generate_step_string(DISTANCE_STEPS, num)
    if USE_PROBABILITY_STEPS:
        res += "\n} else { \n"
        res += __generate_proba_steps(PROBABILITY_STEPS)
    res += "\n}\n}\n"
    return res


def _array_initialize(features):
    fs = [str(e) for e in features]
    if len(fs) == 0:
        return "{}"
    count = 0
    fs_per_line = 10
    res = "{"
    for i, e in enumerate(fs):
        count += 1
        res += e + ","
        if i != len(features) - 1 and count % fs_per_line == 0:
            res += "\n"
    return res[:-1] + "}"


def __generate_step_string(steps, num=None):
    first_step = steps[0]
    res = f"if (distance <= {first_step}) {{\n"
    for i, step in enumerate(steps[1:]):
        res += f"if (distance < {step}) {{\n"
        if WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES:
            res += f'ddd = {i % 7};\n'
            # res += f'printf("branch {i}, ddd=%d\\n", ddd);\n'
            if i == len(steps) - 2:
                res += __step_filename(step, num)
        elif WRITE_AE_ALWAYS_IN_IF:
            res += __step_filename(step, num)
        else:
            if i == len(steps) - 2:
                res += __step_filename(step, num)
    if not WRITE_AE_ALWAYS_IN_IF and not WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES:
        for step in reversed(steps[:-1]):
            res += "} else { \n"
            res += f"{__step_filename(step, num)}"
            res += "}"
    else:
        res += "}" * (len(steps) - 1)
    if WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES:
        res += '\nif (distance < best_ae_dist) {\n'
        res += 'best_ae_dist = distance;\n'
        res += "double new_dist = (double) round(distance*100000.0) / 100000.0;\n"
        res += 'std::stringstream ss;\n'
        res += 'ss << std::fixed << std::setprecision(5) << new_dist;\n'
        res += 'std::string new_dist_str = ss.str();\n'
        res += 'write_if_not_exist(file_base.append(new_dist_str).append(".txt"), fuzzed_features, original_class, ' \
               'fuzzed_class, check_num, probs);\n'
        res += "}\n"
    res += "}"
    return res


def __generate_proba_steps(steps):
    first_step = steps[0]
    res = f"if (proba_original <= {first_step}) {{\n"
    for i, step in enumerate(steps[1:]):
        res += f"if (proba_original < {step}) {{\n"
        res += f"ppp = {i%6};\n"  # To make sure the step is not compiled away
        if i == len(steps) - 2:
            # This will probably never happen. We just need something that is not compiled away.
            res += 'write_if_not_exist(file_base.append("probasmall.txt"), fuzzed_features, ' \
                   'original_class, fuzzed_class, check_num, probs);\n'
    for _ in range(len(steps)):
        res += "}"
    return res


def __generate_proba_steps_special(epsilon):
    res = 'if (probs[0] < probs[1]) {abort();}\n'  # To make sure the sorting works
    if not PROBA_LIMIT_WITHIN_EPSILON:
        epsilon = 1.0
    res += f'if (distance < {epsilon}) {{\n'
    current_step = PROBA_SPECIAL_START_STEP
    step_size = PROBA_SPECIAL_STEP_SIZE
    count = 0
    while current_step - step_size > 0:
        res += f"if (probs[0] - probs[1] < {round(current_step, 3)}) {{\nppp = {count%8};\n"
        current_step -= step_size
        count += 1
    res += 'write_if_not_exist(file_base.append("probasmall.txt"), fuzzed_features, ' \
           'original_class, fuzzed_class, check_num, probs);\n'
    res += '}' * count
    return res + "}"


def __step_filename(step, num=None):
    if FUZZ_ONE_POINT_PER_INSTANCE:
        filename = f'file_base.append("{step}.txt")'
        num = "check_num"
    else:
        filename = f'"check_{num}-{step}.txt"'
    return f'write_if_not_exist({filename}, fuzzed_features, original_class, fuzzed_class, {num}, probs);\n'


def _generate_adversarial_checks_old(x_test, y_test):
    make_or_empty_dir(CHECK_DIR)

    res = ""
    for i, feature_values in enumerate(x_test):
        _write_check_file(i, feature_values)
        expected_outcome = y_test[i]
        func = f"void check_{i}(double fuzzed_features[], int fuzzed_class, double probs[]){{\n"
        # indent = " "*4
        features_per_line = 4
        line = "if ("
        linecount = 0
        num_lines = 0
        for fi, fval in enumerate(feature_values):
            linecount += 1
            line += f"ir(fuzzed_features[{fi}],{fval})"
            if linecount % features_per_line == 0:
                line += "){"
                num_lines += 1
                func += line
                line = "\nif ("
            else:
                if fi < len(feature_values)-1:
                    line += " && "
                else:
                    line += "){"
                    num_lines += 1
                    func += line
        func += f"\nif (fuzzed_class != {expected_outcome}) {{\n"
        func += f"\taf(fuzzed_features,{expected_outcome},fuzzed_class,{i}, probs);\n"
        func += f"if (fuzzed_class == {expected_outcome}) {{\n //should not be possible\n"
        func += "printf(\"not possible %d\", 1);}}"

        func += "}"*num_lines+"}\n"
        res += func
    return res


def _generate_check_function_calls(n: int):
    if FUZZ_ONE_POINT_PER_INSTANCE:
        return "\ncheck_for_ae(fuzzed_features, fuzzed_class, probs);\n"
    else:
        res = "\n"
        for i in range(n):
            res += f"\tcheck_{i}(fuzzed_features, fuzzed_class, probs);\n"
        return res


def _generate_threshold_init(num_features):
    res = ""
    for i in range(num_features):
        res += f"thresholds_per_feature.push_back(th_f_{i});\n"
    return res


def _generate_threshold_vectors(sorted_threshold_per_feature):
    res = ""
    for i, sorted_thresholds in enumerate(sorted_threshold_per_feature):
        res += f"vector<double> th_f_{i} = {_array_initialize(sorted_thresholds)};\n"
    return res


def _write_check_file(n, fs):
    fs = [str(f) for f in fs]
    feature_string = ",".join(fs)
    with open(join(CHECK_DIR, str(n)), 'w+') as file:
        file.write(feature_string)


def read_check_files():
    files = glob.glob(join(CHECK_DIR, "*"))
    res = dict()
    for f in files:
        with open(f, 'r') as file:
            contents = file.read()

        f = f.replace(CHECK_DIR, "")
        f = f.replace("/", "")

        check_id = int(f)
        fs = contents.split(',')
        float_fs = [float(feat) for feat in fs]
        res.update({check_id: float_fs})
    return res
