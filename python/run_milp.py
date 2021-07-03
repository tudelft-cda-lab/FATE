import json
import statistics
import time
import numpy as np
from datetime import datetime
from constants import NUM_THREADS, NUM_ADV_CHECKS
from datasets import is_difficult_for_milp, get_num_classes
from external.kantchelian_attack import KantchelianAttackWrapper
from run_fate import generate_model_path, generate_x_test_path, generate_y_test_path, take_n
from run_zhang import _generate_json_filename
from utils import load, to_json, write_file

# _datasets = ['WEB', 'VOWEL']
# _datasets = ['MNIST', 'FMNIST']
_datasets = ['BC', 'DIABETES', 'IJ', 'COV', 'HIGGS', '26']
_mts = ['GB', 'RF']
_output_filename = 'milp_results.csv'
n_iter = 1


def run_all():
    out_str = "Model Type,Dataset,Avg Norm,Avg Time,Time dev,Num points attacked,Float mis-classifications\n"
    for dataset_name in _datasets:
        num_classes = get_num_classes(dataset_name)
        for mt in _mts:

            classifier = load(generate_model_path(dataset_name, mt))

            x_test = load(generate_x_test_path(dataset_name, mt))
            y_test = load(generate_y_test_path(dataset_name, mt))

            if is_difficult_for_milp(dataset_name):
                nc = 50
            else:
                nc = NUM_ADV_CHECKS
            x_test, y_test = take_n(classifier, x_test, y_test, nc)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            json_filename = _generate_json_filename(dataset_name, mt)
            to_json(classifier, json_filename, mt)
            # pred_thres = 0.0
            
            # if mt == 'GB':
            #     y_train = load(generate_y_train_path(dataset_name, mt))
            #
            #     num_classes = get_num_classes(dataset_name)
            #
            #     initial_prediction = predict_initial(y_train, num_classes)
            #     pred_thres -= initial_prediction/DEFAULT_LEARNING_RATE_GB

            avg_dist = 0
            num_float_misc = 0
            avg_times = []
            for num_iter in range(n_iter):
                print("Iteration ", num_iter)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time = ", current_time)

                t_0 = time.time()
                json_model = json.load(open(json_filename, "r"))
                opts = {
                    "epsilon": None,
                    "round_digits": 5,
                    "pred_threshold": 0.0,
                    "low_memory": True,
                    "verbose": True,
                    "n_threads": NUM_THREADS,
                }
                aw = KantchelianAttackWrapper(json_model, num_classes)
                distances = aw.attack_distance(x_test, y_test, options=opts)
                t = time.time() - t_0
                print(distances)
                avg_dist = np.mean(distances) if len(distances) > 1 else distances[0]
                avg_time = t/len(distances)
                # avg_dist, avg_time, adversarial_features = attack_json_for_X_y(json_filename,
                #                                                                x_test,
                #                                                                y_test,
                #                                                                n_classes=num_classes,
                #                                                                verbose=is_difficult_for_milp(
                #                                                                    dataset_name),
                #                                                                n_threads=NUM_THREADS,
                #                                                                pred_threshold=0.0,
                #                                                                round_digits=5)
                #
                # real_predictions = classifier.predict(adversarial_features)
                #
                # num_float_misc = 0
                # for i, label in enumerate(y_test):
                #     ae_pred = real_predictions[i]
                #     if label == ae_pred:
                #         # original = x_test[i]
                #         # fuzzed = adversarial_features[i]
                #         # print('Misclassified')
                #         # print('Original: ', original)
                #         # print('Fuzzed: ', fuzzed)
                #         # sum_val_fuzzed = decisions[i]
                #         # sum_val_original = decisions_original[i]
                #         # if sum_val_original < sum_val_fuzzed:
                #         #     print('Decision: ', sum_val_fuzzed)
                #         #     print('Decision original: ', sum_val_original)
                #         # print('Diff: ', np.abs(original-fuzzed))
                #         num_float_misc += 1

                avg_times.append(avg_time)

            num_points = len(y_test)
            mean_time = statistics.mean(avg_times) if n_iter > 1 else avg_times[0]
            std_time = statistics.stdev(avg_times) if n_iter > 1 else 0
            print(f'{mt}-{dataset_name}: avg-norm = {avg_dist}, avg-time = {mean_time} (+/- {std_time}),'
                  f'num-attacked: {num_points}', f'float mis-classifications: {num_float_misc}')

            out_str += f'{mt},{dataset_name},{avg_dist},{mean_time},{std_time},{num_points},{num_float_misc}\n'
    write_file(_output_filename, out_str)


if __name__ == '__main__':
    run_all()
