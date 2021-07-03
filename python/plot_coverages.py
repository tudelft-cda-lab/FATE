import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants import DISTANCE_NORM


COV_VS_DIST = False
COV_VS_POINTS_ATT = True


def plot_lines(dataset_name, epsilon, mt):
    x, y_norm, y_attacked, y_coverage, dist_own, dist_other = _read_file(dataset_name, epsilon, mt, False)
    sns.lineplot(x=x, y=y_norm, estimator='mean', label=f'Avg {DISTANCE_NORM} distance')
    sns.lineplot(x=x, y=y_attacked, estimator='mean', label='Attacked (%)')
    sns.lineplot(x=x, y=y_coverage, estimator='mean', label='Coverage (%)')
    sns.lineplot(x=x, y=dist_own, estimator='mean', label=f'Avg distance to own class')
    ax = sns.lineplot(x=x, y=dist_other, estimator='mean', label=f'Avg distance to other classes')
    ax.set(xlabel='Number of runs', title=f'Development of Adversarial Examples for {dataset_name} - {epsilon} ({mt})')

    # plt.plot(x, y_norm, label='Avg l-inf')
    # plt.plot(x, y_attacked, label='Attacked (%)')
    # plt.plot(x, y_coverage, label='Coverage (%)')
    # plt.plot(x, dist_own, label='Avg distance to own class')
    # plt.plot(x, dist_other, label='Avg distance to other class')
    # plt.title()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter(dataset_name, epsilon, mt):
    x, y_norm, y_attacked, y_coverage, dist_own, dist_other = _read_file(dataset_name, epsilon, mt, True)
    # filtered_y_n, filtered_y_a, filtered_y_c = [], [], []
    # for xx, y_n, y_a, y_c in zip(x, y_norm, y_attacked, y_coverage):
    #     if xx == 200000:
    #         filtered_y_a.append(y_a)
    #         filtered_y_n.append(y_n)
    #         filtered_y_c.append(y_c)

    if COV_VS_DIST:
        ax = sns.regplot(x=y_norm, y=y_coverage, fit_reg=True, scatter=True)
        ax.set(xlabel=f'Avg {DISTANCE_NORM} distance', ylabel='Coverage (%)',
               title=f'Avg {DISTANCE_NORM} distance vs Coverage for {dataset_name} - {epsilon} ({mt})')
        # z = np.polyfit(y_norm, y_coverage, 1)
    elif COV_VS_POINTS_ATT:
        ax = sns.regplot(x=y_attacked, y=y_coverage, fit_reg=True, scatter=True)
        ax.set(xlabel=f'Points attacked (%)', ylabel='Coverage (%)',
               title=f'Points attacked vs Coverage for {dataset_name} - {epsilon} ({mt})')
        # z = np.polyfit(y_attacked, y_coverage, 1)
    else:
        raise ValueError('Do not know what to plot in the scatter plot')

    # plt.scatter(y_norm, y_coverage)
    # p = np.poly1d(z)
    # plt.plot(y_norm, p(y_norm))
    plt.gca().set_aspect('equal')
    # plt.plot(x, y_attacked, label='Attacked (%)')
    # plt.plot(x, y_coverage, label='Coverage (%)')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def _read_file(dataset_name, epsilon, mt, scatter):
    from run_fate import generate_coverages_path
    with open(generate_coverages_path(dataset_name, epsilon, mt, scatter)) as file:
        lines = file.readlines()

    x = []
    y_norm = []
    y_attacked = []
    y_coverage = []
    dist_own = []
    dist_other = []
    for line in lines:
        runs, avg_best_norm, perc_attacked, avg_coverage, avg_fuzzing_time, avg_dist_to_own_class, \
            avg_dist_to_other_class = line.split(",")
        runs = int(runs)
        avg_best_norm = float(avg_best_norm)
        perc_attacked = float(perc_attacked)
        avg_coverage = float(avg_coverage)
        avg_dist_to_own_class = float(avg_dist_to_own_class)
        avg_dist_to_other_class = float(avg_dist_to_other_class)
        x.append(runs)
        y_norm.append(avg_best_norm)
        y_attacked.append(perc_attacked)
        y_coverage.append(avg_coverage)
        dist_own.append(avg_dist_to_own_class)
        dist_other.append(avg_dist_to_other_class)
    return x, y_norm, y_attacked, y_coverage, dist_own, dist_other


if __name__ == '__main__':
    # plot_lines('BC', 0.3, 'GB')
    plot_scatter('BC', 0.1, 'GB')
