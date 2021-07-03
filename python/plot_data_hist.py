from os.path import join

from constants import IMAGE_DIR
from datasets import DATASETS, get_full_name, get_num_classes, get_num_features, get_dataset_description, get_link
from run_fate import load_data
import matplotlib.pyplot as plt
import numpy as np

from utils import write_file


def run():
    # count = 0
    out_str = "\\chapter{Datasets}\n\\label{app:data}\n"
    out_str += "The following datasets were mainly selected because they are were used in related work \cite{zhang2020efficient, chen2019robustness}. They are a combination of binary and multi-class datasets and have varying amounts of features and model sizes. The Breast-cancer and Diabetes datasets are very small and generally easy to attack. MNIST 2vs 6, MNIST and FMNIST are image recognition datasets and have a lot of features (784). They have the largest models and are generally the most difficult to attack.\n"
    for dataset in DATASETS:
        print(dataset)
        _, _, y_train, y_test = load_data(dataset)
        # y_train = sorted(y_train)
        # y_test = sorted(y_test)
        train_vals, train_counts = np.unique(y_train, return_counts=True)
        test_vals, test_counts = np.unique(y_test, return_counts=True)
        full_name = get_full_name(dataset)
        # sns.histplot(y_test)
        # sns.barplot(x=train_vals, y=train_counts, label='Training-set')
        # sns.barplot(x=test_vals, y=test_counts, label='Test-set')
        fig = plt.figure()
        plt.bar(train_vals-0.2, train_counts, width=0.4, label='Training-set')
        plt.bar(test_vals+0.2, test_counts, width=0.4, label='Testing-set')
        plt.xticks(train_vals)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        # plt.title(f"Distribution of class labels for {full_name}")
        plt.tight_layout()
        fig.savefig(join(IMAGE_DIR, f'hist_{dataset}.png'))

        num_classes = get_num_classes(dataset)
        num_features = get_num_features(dataset)
        num_train = len(y_train)
        num_test = len(y_test)
        description = get_dataset_description(dataset)
        link = get_link(dataset)
        link_str = f"Link: \\url{{{link}}}. " if link is not None else ""
        fig_label = f'fig:data_hist_{dataset}'

        out_str += f"""
\\section{{{full_name}}}
{link_str}{description} The dataset consists of {num_train} samples in the training set and {num_test} samples in the testing set. There are {num_classes} classes and each sample has {num_features} features. 
The distribution of samples among the classes can be seen in \\autoref{{{fig_label}}}. 
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=.4\\linewidth]{{img/data_hist/hist_{dataset}.png}}
    \\caption{{Distribution of class labels for {full_name}}}
    \\label{{{fig_label}}}
\\end{{figure}}\n\n
"""
        # plt.show()
        # count += 1
        # if count == 1:
        #     break
    write_file('data_hist_latex.txt', out_str)
    print(out_str)


if __name__ == '__main__':
    run()
