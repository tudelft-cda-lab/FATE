import os
from os.path import join

INVESTIGATE = False  # Records coverages and saves them. Generates a plot in the end. Do not use with automate.
TEST_OUTSIDE_FUZZER = False  # Runs FATE as standalone (1+1) EA
BLACKBOX = False  # Disables white-box information such as thresholds and feat imp.
FORCE_DEFAULT_EPSILON = True or TEST_OUTSIDE_FUZZER  # Runs all datasets with the default epsilon
FORCE_DEFAULT_MUTATION_CHANCE = False or TEST_OUTSIDE_FUZZER  # Runs all datasets with the default mutation chance
LIMIT_TIME = True  # If false, run 10 times as long

############ FATE Standalone ############
CROSSOVER_CHANCE = 0.001  # Chance that crossover occurs
CROSSOVER_RANDOM_CHANCE = 1.0  # Actual chance for crossover with random features is 0.001
# CROSSOVER_CHANCE * CROSSOVER_RANDOM_CHANCE
NUM_RUNS = 100000000  # Unlimited. Change for smaller amount of runs
POPULATION_SIZE = 1  # Population size.

############ RQ 1 defaults ############
MEASURE_EXEC_P_S = True  # Parse the number of executions per second.
ALLOW_FLOAT_MIS_CLASSIFICATION = True  # If True, do not filter mis-classifications from the produced AE
CONSISTENT_DRAWS = True  # Seeds random with 0, to create consistent check-set draws
FUZZ_ONE_POINT_PER_INSTANCE = True  # compile to generic fuzz target and fuzz per point
USE_CUSTOM_MUTATOR = True  # If False, use the standard mutator of LibFuzzer
USE_CROSSOVER = True and USE_CUSTOM_MUTATOR  # Combines mutation with crossover (split at random location)
USE_GAUSSIAN = True  # Gaussian vs random uniform mutation
USE_PROBABILITY_STEPS_SPECIAL = True  # Proba descent based on small proba diff between 2nd class predicted
PROBA_LIMIT_WITHIN_EPSILON = True  # Only save seeds if within epsilon
WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES = True  # Saves execution time
ALWAYS_OPTIMIZE = True  # Otherwise only optimize small files
MUTATE_DEPTH = 7 if TEST_OUTSIDE_FUZZER else 5  # The maximum number of consecutive mutations per seed for LibFuzzer
DEFAULT_EPSILON = 0.1 if TEST_OUTSIDE_FUZZER else 0.2  # Default epsilon
DEFAULT_MUTATE_CHANCE = 0.5 if TEST_OUTSIDE_FUZZER else 0.1  # Chance that a single features is mutated
FUZZER = 'libFuzzer'
# FUZZER = 'AFL++'
# FUZZER = 'honggfuzz'
# FUZZER = 'AFLGo'
FUZZERS = ['libFuzzer', 'AFL++', 'AFLGo', 'honggfuzz']
if FUZZER not in FUZZERS:
    raise ValueError(f'Fuzzer {FUZZER} not recognised, should be one of [{", ".join(FUZZERS)}]')
if FUZZER == 'honggfuzz' and USE_CUSTOM_MUTATOR:
    raise ValueError('Honggfuzz and custom mutator is not supported')


############ RQ 2 defaults ############
AE_MUTATE_TOWARDS_VICTIM = True  # If AE, mutate values only towards victim point.
MUTATE_BIGGEST_CHANCE = 0.5  # When an AE is found, the chance to only mutate all biggest difference fs towards victim
ALSO_MUTATE_BIGGEST = True  # Always mutate all features > the biggest l-inf distance - 0.01. Only with FUZZ_ONE
# These alter the chance that a feature is mutated
BIAS_MUTATE_BIG_DIFFS = True
USE_THRESHOLDS_FOR_MUTATION = True and not BLACKBOX  # move to optimal boundary value after drawing from mutation dist
# Fuzzes for each datapoint with and without AE init
DOUBLE_FUZZ_WITH_AE = True and not (TEST_OUTSIDE_FUZZER or INVESTIGATE)
USE_FEATURE_IMPORTANCE = True and not BLACKBOX  # prioritize more important features for mutation
INITIALIZE_WITH_POINT_IN_BETWEEN = True and DOUBLE_FUZZ_WITH_AE
INITIALIZE_WITH_EXTRA_POINTS_IN_BETWEEN = True and INITIALIZE_WITH_POINT_IN_BETWEEN

if TEST_OUTSIDE_FUZZER and (not FUZZ_ONE_POINT_PER_INSTANCE):
    raise ValueError('Test outside fuzzer conflicting options')
if TEST_OUTSIDE_FUZZER and DOUBLE_FUZZ_WITH_AE and (POPULATION_SIZE < 2 or CROSSOVER_RANDOM_CHANCE > 0.99):
    raise ValueError('Test outside fuzzer double fuzz configuration problem')

############ RQ 1.2 defaults ############
FILTER_BAD_AE = True  # If True, discards all AE that are worse than FAILURE_THRES
FUZZ_ONLY_COV_FOR_FOREST = False  # Only insert coverage-guidance for the lines that belong to the Forest
FUZZ_ONLY_COV_FOR_CHECK = True  # Only insert coverage-guidance for the lines that belong to the objective function
FUZZ_WITHOUT_COVERAGE_GUIDANCE = False  # If True, baseline: removes almost all coverage guidance (except TestOneInput)
if FUZZER == 'AFL++' and FUZZ_WITHOUT_COVERAGE_GUIDANCE:
    raise ValueError('AFL++ crashes because the fuzzer name cannot be set with the -n (no instrument) option')

############ Objective function settings ############
COMBINE_DISTANCE_AND_PROBABILITY = False  # distance = distance + probability
USE_PROBABILITY_STEPS = False  # probability steps in the check function ELSE branch
PROBA_SPECIAL_ALWAYS = False
PROBA_SPECIAL_START_STEP = 0.2
PROBA_SPECIAL_STEP_SIZE = 0.01
WRITE_AE_ALWAYS_IN_IF = False  # Slower option for the objective function
if USE_PROBABILITY_STEPS and USE_PROBABILITY_STEPS_SPECIAL:
    raise ValueError('Select at most one type of probability step')
if WRITE_AE_ALWAYS_IN_IF and WRITE_AE_ONLY_IF_BETTER_OUTSIDE_BRANCHES:
    raise ValueError('Only one write_X can be used on the settings')

############ Fuzzer settings ############
NEVER_OPTIMIZE = False
FORCE_ENTROPIC = False  # libfuzzer. Experimental. Enables entropic power schedule.
NO_ENTROPIC = False
FOCUS_FUNCTION = "0"  # focus_function 0 Experimental. Fuzzing will focus on inputs that trigger calls
# # to this function. If -focus_function=auto and -data_flow_trace is used, libFuzzer will choose the
# focus functions automatically.
if sum([FUZZ_WITHOUT_COVERAGE_GUIDANCE, FUZZ_ONLY_COV_FOR_CHECK, FUZZ_ONLY_COV_FOR_FOREST]) > 1:
    raise ValueError('Only one coverage guidance option can be used at the same time')
if NEVER_OPTIMIZE and ALWAYS_OPTIMIZE:
    raise ValueError('Conflicting optimize options')

############ AFL settings ############
# TIME_NO_NEW_COV = 10
IS_AE_CHANCE = 0.5  # Because we cannot access the fuzzer logic in the mutator
NUM_CYCLES_IN_LOOP = 1000  # Number of consecutive iterations after which we start with a clean sheet
AFL_USE_DICT = True and not USE_CUSTOM_MUTATOR
AFL_USE_CMP_LOG = False and not USE_CUSTOM_MUTATOR
ENABLE_DETERMINISTIC = False
SKIP_DETERMINISTIC = False
# see docs/power_schedules.md
AFL_SCHEDULE = None  # one of fast(default, use None), explore, exploit, seek, rare, mmopt, coe, lin, quad
# AFL generic
AFL_MUTATE_FILENAME = "afl_mutation.cc"
AFL_OUTPUT_DIR = "afl_out"
# AFL++
AFLPP_DICT_PATH = join(os.getcwd(), 'afl_dict')
AFLPP_TEMPLATE_PATH = "templates/aflpp.jinja2"
MUTATE_TEMPLATE_PATH = "templates/mutate.jinja2"
AFLPP_COMPILER_PATH = "afl-clang-lto++"
# AFLPP_COMPILER_PATH = "afl-clang-fast++"
# AFLGo
AFL_GO_COMPILER_PATH = "/home/cas/AFLGo/afl-clang-fast++"
AFL_GO_FUZZ_PATH = "/home/cas/AFLGo/afl-fuzz"
AFL_GO_GEN_DIST_PATH = "/home/cas/AFLGo/scripts/gen_distance_fast.py"
AFL_GO_TARGETS_FILE = 'BBtargets.txt'
AFLGO_TEMPLATE_PATH = "templates/aflgo.jinja2"

############ honggfuzz settings ############
HONG_COMPILER_PATH = "/home/cas/honggfuzz/hfuzz_cc/hfuzz-clang++"
HONG_FUZZER_PATH = "/home/cas/honggfuzz/honggfuzz"
HONG_OUTPUT_DIR = "hongg_out"

############ Mutation settings ############
MINIMIZE_THRESHOLD_LIST = False  # Removes all thresholds within 0.0001 from each other
IS_AE_FAKE = False  # Fakes the model query if the current input is an AE
USE_WAS_AE = False  # Saves the result of the last known model query
STEEP_CURVE = False  # If True, square the draw from the gaussian distribution, such that smaller draws are more likely
# feature importance is calculated by its occurrence
FEATURE_IMPORTANCE_BASED_ON_OCCURRENCE = False and USE_FEATURE_IMPORTANCE
MUTATE_LESS_WHEN_CLOSER = False  # When True, multiplies mutation with largest diff between fuzzed and victim.
# as splitting threshold in the forest. Cannot be true together with AE_MUTATE_TOWARDS_VICTIM
AE_CHECK_IN_MUTATE = (ALSO_MUTATE_BIGGEST or BIAS_MUTATE_BIG_DIFFS or USE_THRESHOLDS_FOR_MUTATION or
                      AE_MUTATE_TOWARDS_VICTIM or MUTATE_LESS_WHEN_CLOSER) and FUZZ_ONE_POINT_PER_INSTANCE \
                     and FUZZER != 'AFL++'

if MUTATE_LESS_WHEN_CLOSER and AE_MUTATE_TOWARDS_VICTIM:
    raise ValueError('Mutate less and AE mutate towards original cannot be used together')

############ AE init ############
# k-ANN structure
ANN_TREES = 10  # the amount of trees for the "annoy" lookup
K_ANN = 10  # how many nearest neighbours to find
NO_SEED_INIT = False  # When True, each run is only seeded with all-0 features. No input is not possible, because
# The custom mutator would otherwise break.
INITIALIZE_WITH_AE = False  # use ANN to seed with K_ANN closest data-points from other classes
INITIALIZE_WITH_AVG_OPPOSITE = False  # For binary-classification: seed with average member of the other class
INITIALIZE_WITH_POINT_IN_BETWEEN = INITIALIZE_WITH_POINT_IN_BETWEEN or \
                                   (True and INITIALIZE_WITH_AE)
INITIALIZE_WITH_EXTRA_POINTS_IN_BETWEEN = INITIALIZE_WITH_EXTRA_POINTS_IN_BETWEEN or \
                                          (True and INITIALIZE_WITH_POINT_IN_BETWEEN)

INITIALIZE_WITH_FULL_TRAIN_SET = False  # Put all instances of other class from test set in corpus.
if INITIALIZE_WITH_FULL_TRAIN_SET and (INITIALIZE_WITH_AE or DOUBLE_FUZZ_WITH_AE):
    raise ValueError('INITIALIZE_WITH_FULL_TRAIN_SET cannot be used with INITIALIZE_WITH_AE or DOUBLE_FUZZ_WITH_AE')
if sum([INITIALIZE_WITH_AE, INITIALIZE_WITH_AVG_OPPOSITE, INITIALIZE_WITH_FULL_TRAIN_SET]) > 1:
    raise ValueError('Conflicting initialize options')

############ Testing ############
DEBUG = False  # If True, shows output and runs 1 sample with 1 thread only.
MEASURE_COVERAGE = False  # Measure coverage through instrumentation, costs exec/s
SKIP_COMPILATION = False
COMPILE_ONLY = False
PRINT_NUMBER_OF_LEAVES = False  # Estimate for model size
INVESTIGATE_WITH_SCATTER = False and INVESTIGATE  # Shows a scatter plot instead of a line plot when INVESTIGATE
NUM_INVESTIGATE_RUNS = 5  # The number of repetitions for creating plots.
FAILURE_THRES = 0.9  # See FILTER_BAD_AE
SHOW_OUTPUT = False or DEBUG  # Shows fuzzer output
CREATE_LOOKUP = False or INITIALIZE_WITH_AE or INITIALIZE_WITH_AVG_OPPOSITE or INVESTIGATE \
                or INITIALIZE_WITH_FULL_TRAIN_SET or DOUBLE_FUZZ_WITH_AE
if DEBUG and MEASURE_EXEC_P_S:
    raise ValueError('Debug and measure exec/s cannot be used at the same time')
if INVESTIGATE and DOUBLE_FUZZ_WITH_AE:
    raise ValueError('Double fuzz together with investigate should not be used.')

NUM_DEBUG = 1
NUM_THREADS = 10 if not DEBUG else NUM_DEBUG  # Number of simultaneous fuzzing instances, but is also
# Used for training the ensembles, the MILP attack and the lt-attack (Zhang)
NUM_ADV_SUPER_QUICK = 10  # The number of victims to attack for runs with the -qq flag.
NUM_ADV_QUICK = 50  # The number of victims to attack for runs with the -q flag.
NUM_ADV_CHECKS = 500 if not DEBUG else NUM_DEBUG  # number of adversarial victims
MAX_POINTS_LOOKUP = 5000  # The AE lookup will be created over this amount of training samples maximum
DEFAULT_TIME_PER_POINT = 1  # The default fuzzing time per datapoint

MODEL_TYPES = ['RF', 'GB']  # the identifiers of the model types (Random Forest, Gradient Boosting)
DISTANCE_NORMS = ['l_0', 'l_1', 'l_2', 'l_inf']
DISTANCE_NORM = 'l_inf'  # the norm to calculate the distance in the fuzzer
if DISTANCE_NORM not in DISTANCE_NORMS:
    raise ValueError(f'Norm {DISTANCE_NORM} not recognised, should be one of [{", ".join(DISTANCE_NORMS)}]')

DISTANCE_STEPS = [round(0.005 * i, 3) for i in reversed(range(1, 201))]  # [1.0, 0.995, ..., 0.005]
# DISTANCE_STEPS = [round(0.001 * i, 3) for i in reversed(range(1, 1001))]  # [1.0, 0.999, ..., 0.001]
# DISTANCE_STEPS = [round(0.01 * i, 2) for i in reversed(range(1, 101))]  # [1.0, 0.99, ..., 0.01]
# DISTANCE_STEPS = [round(0.1 * i, 1) for i in reversed(range(1, 11))]  # [1.0, 0.99, ..., 0.01]
# DISTANCE_STEPS = [0.8, 0.7, 0.6] \
#                  + [round(0.01 * i, 2) for i in reversed(range(11, 51))] \
#                  + [round(0.001 * i, 3) for i in reversed(range(1, 101))]  # Decreasing
# DISTANCE_STEPS = [0.8, 0.7] \
#                  + [round(0.01 * i, 2) for i in reversed(range(25, 70))] \
#                  + [round(0.001 * i, 3) for i in reversed(range(20, 250))] \
#                  + [round(0.0001 * i, 4) for i in reversed(range(1, 200))]  # Decreasing very small
DISTANCE_STEPS.append(0.000001)
PROBABILITY_STEPS = [round(0.01 * i, 2) for i in reversed(range(1, 51))]  # [1.0, 0.99, ..., 0.01]
# PROBABILITY_STEPS = [0.8, 0.7, 0.6] \
#                  + [round(0.01 * i, 2) for i in reversed(range(1, 51))]  # [0.8, 0.7, ..., 0.5, 0.49...]
# PROBABILITY_STEPS = [round(0.5 + 0.05 * i, 2) for i in reversed(range(1, 11))] \
#     + [round(0.2 + 0.01 * i, 2) for i in reversed(range(1, 31))] \
#     + [round(0.005 * i, 3) for i in reversed(range(1, 41))]

# Directories, all relative to main folder (code)
CHECK_DIR = "python/.CHECK"
IMAGE_DIR = 'python/img'
RESULTS_DIR = "python/.RESULTS"
COVERAGES_DIR = "python/.COVERAGES"
MODEL_DIR = "python/models"
JSON_DIR = join(MODEL_DIR, 'json')
DATA_DIR = "python/data"
LIB_SVM_DIR = join(DATA_DIR, 'libsvm')
OPEN_ML_DIR = join(DATA_DIR, 'openml')
ZHANG_DATA_DIR = join(DATA_DIR, 'zhang')
CORPUS_DIR = ".GENERATED_CORPUS"
ADV_DIR = ".ADVERSARIAL_EXAMPLES"
ZHANG_CONFIG_DIR = ".ZHANG_CONFIGS"

# Files
NUM_FEATURES_PATH = ".num_features"
LIBFUZZER_TEMPLATE_PATH = "templates/libfuzzer.jinja2"
OUTPUT_FILE = "fuzzme.cc"

# WARNING, run with --reload (once for each dataset) after changing these.
DEFAULT_LEARNING_RATE_GB = 0.1
TEST_FRACTION = 0.2
NUM_SAMPLES = 2500  # For synthetic datasets

# Better not change these
SMALL_PERTURBATION_THRESHOLD = 0.00001
THRESHOLD_DIGITS = 7
BYTES_PER_FEATURE = 8  # float = 4, double = 8
TIME_PRECISION = 4


def get_num_adv(): return NUM_ADV_CHECKS
