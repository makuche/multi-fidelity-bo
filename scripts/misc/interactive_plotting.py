import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import read_write

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent/'data/processed'


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 interactive_plotting.py <experiment> <measure>\n \
        Example: python3 interactive_plotting.py 2LFbasic1 gmp")
        return 1
    experiment_path = PROCESSED_DIR.joinpath(sys.argv[1])
    measure = sys.argv[2]
    experiment_runs = [exp for exp in
                       experiment_path.iterdir() if exp.is_file()]
    data = [read_write.load_json(run, '') for run in experiment_runs]
    data.sort(key=sort_data_by_convergence)
    plot_functions = [
        plot_amplitude, plot_xy, plot_hyperparameter, plot_best_acq,
        plot_gmp_prediction, plot_xhat, plot_gmp_statistics]

    for plot_function in plot_functions:
        if measure in plot_function.__name__:
            fig, ax = plt.subplots(figsize=(18, 12))
            plot_function(data, fig, ax)


def sort_data_by_convergence(data):
    if data['iterations_to_gmp_convergence'][5] is None:
        return np.infty
    else:
        return data['iterations_to_gmp_convergence'][5]


def plot_amplitude(data, fig, ax):
    for exp_run_idx, exp_run in enumerate(data):
        xy = np.array(data[exp_run_idx]['xy'])
        amps = [0.5*(max(xy[:xy_idx, -1]) - min(xy[:xy_idx, -1]))
                for xy_idx in range(1, xy.shape[0])]
        convergence = exp_run['iterations_to_gmp_convergence'][5]
        label = f'run {exp_run_idx+1}: convergence at {convergence}'
        plt.plot(np.arange(len(amps)), amps, label=label)

    plt.xlabel(r'Iteration $n$')
    plt.ylabel(r'GMP')
    plt.legend()
    plt.title('Amplitude over iteration')
    plt.show()


def plot_xy(data, fig, ax):
    pass


def plot_hyperparameter(data, fig, ax):
    pass


def plot_best_acq(data, fig, ax):
    pass


def plot_gmp_prediction(data, fig, ax):
    longest_run_duration = 0
    tol = 0.1
    for exp_run_idx, exp_run in enumerate(data):
        gmp = np.array(data[exp_run_idx]['gmp'])
        longest_run_duration = max(longest_run_duration, gmp.shape[0])
        convergence = exp_run['iterations_to_gmp_convergence'][5]
        plt.plot(np.arange(gmp.shape[0]), gmp[:, -2])
        label = f'run {exp_run_idx+1}: convergence at {convergence}'
        plt.scatter(np.arange(gmp.shape[0]), gmp[:, -2], s=10, label=label)

    plt.axhline(tol, alpha=.3, color='k', linestyle='dashed',
                label='0.1 kcal/mol tolerance')
    plt.axhline(-tol, alpha=.3, color='k', linestyle='dashed')
    plt.ylim(-1, 3)
    plt.ylabel('GMP')
    plt.xlabel(r'Iteration $n$')
    plt.legend()
    plt.title('GMP over iteration')
    plt.show()


def plot_gmp_statistics(data, fig, ax):
    tol = 0.1
    convergences = np.zeros(len(data))
    gmps = np.zeros((len(data), len(data[0]['gmp'])))
    for exp_run_idx, exp_run in enumerate(data):
        gmps[exp_run_idx, :] = np.array(data[exp_run_idx]['gmp'])[:, -2]
        convergences[exp_run_idx] = exp_run['iterations_to_gmp_convergence'][5]

    gmp_mean = np.mean(gmps, axis=0)
    gmp_var = np.var(gmps, axis=0)
    gmp_median = np.median(gmps, axis=0)
    x_range = np.arange(gmp_mean.shape[0])
    plt.plot(x_range, gmp_mean, label=r'mean($y$)')
    plt.plot(x_range, gmp_median, label=r'median($y$)')
    plt.plot(x_range, gmp_mean + 2*gmp_var, linestyle='--', c='k',
             label=r'mean($y$) $\pm 2\sigma$')
    plt.plot(x_range, gmp_mean - 2*gmp_var, linestyle='--', c='k')
    plt.fill_between(x_range, gmp_mean - 2*gmp_var, gmp_mean + 2*gmp_var,
                     alpha=.1, color='k')
    plt.axhline(tol, alpha=.3, color='k', linestyle='dashed',
                label='0.1 kcal/mol tolerance')
    plt.axhline(-tol, alpha=.3, color='k', linestyle='dashed')
    plt.ylim(-1, 3)
    plt.legend(fontsize=15)
    plt.title(f'GMP over iteration for {len(data)} run/s')
    plt.show()


def plot_xhat(data, fig, ax):
    pass


if __name__ == '__main__':
    main()
