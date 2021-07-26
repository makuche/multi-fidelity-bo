import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import read_write


verbose = False
show_plots = False

SMALL_SIZE = 12
MEDIUM_SIZE = 20
LARGE_SIZE = 30

THESIS_DIR = Path(__name__).resolve().parent.parent.parent
CONFIG = read_write.load_yaml(
    THESIS_DIR.joinpath('scripts/analyse/config'), '/statistics.yaml')
exp_list = [THESIS_DIR / 'data' / 'processed' /
            exp for exp in CONFIG['sobol']]
#NAMES = ['UHF', 'UHF0', 'HF', 'LF']
NAMES = ['UHF', 'HF', 'LF']

def main():
    y_values = plot_y_scatter_trellis(exp_list)
    acq_times = plot_acq_times_comparison(exp_list)
    plot_acq_times_histograms(acq_times, NAMES)
    correlation_matrix = calculate_correlation_matrix(y_values)
    summary_statistics = calculate_summary_statistics(y_values)

def plot_y_scatter_trellis(exp_list, figname='trellis_correlation.pdf',
                           data_points=100):
    """Create scatter trellis plot of sobol que experiments.

    Args:
        exp_list (list): List containing the experiments .json file paths
        figname (str, optional): File name for figure. Defaults to
        'linear_correlation.pdf'.
    """
    N = len(exp_list)

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    y_values = np.zeros((N, data_points))
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir()]
        exp_runs.sort()
        for exp_idx, exp_run in enumerate(exp_runs):
            data = read_write.load_json(exp_run, '')
            if len(exp_runs) > 1:
                # idxs sets the correct indexes to fill in y_values array
                if 'exp_5' in str(exp_run):
                    idxs = (20*exp_idx, 19 + 20*exp_idx)
                elif 'exp_6' in str(exp_run):
                    idxs = (98, 100)
                else:
                    idxs = (20*exp_idx, 20 + 20*exp_idx)
                y_values[idx, idxs[0]:idxs[1]] = np.array(data['xy'])[:,-1]
            else:
                y_values[idx,:] = np.array(data['xy'])[:data_points,-1]

    fig, axs = plt.subplots(N,N, figsize=(5*N,5*N), constrained_layout=True)
    for i in range(N):
        ax = axs[i,i]
        ax.axis('off')
        ax.text(0.5, 0.5, f'{NAMES[i]}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=45,
            transform=ax.transAxes)
        for j in range(i+1,N):
            ax = axs[i,j]
            axs[i,j].scatter(y_values[j,:], y_values[i,:], marker='x', \
                color='blue', alpha=.5)
            ax.set_xticks(axs[0,1].get_yticks())
            ax.set_yticks(axs[0,1].get_xticks())
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis = 'both',
              width = 3, length = 4)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                   ]

            # now plot both limits against eachother
            ax.plot(lims, lims, linestyle='dashed', alpha=0.25, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        # remove below-diagonal plots
        for j in range(i):
            ax = axs[i,j]
            ax.axis('off')
    axs[0,N-1].set_xticklabels([])
    axs[0,N-1].set_yticklabels([])
    plt.savefig(''.join(('../../results/figs/', figname)))
    plt.close()
    return y_values


def plot_acq_times_comparison(exp_list, figname='acquisition_times.pdf',
                              num_points=100):

    N = len(exp_list)

    if N == 4:
        font = {'size': 16}
    else:
        font = {'size': 20}
    plt.rc('font', **font)

    acq_times = np.zeros((N, num_points))
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir()]
        exp_runs.sort()
        for exp_idx, exp_run in enumerate(exp_runs):
            data = read_write.load_json(exp_run, '')
            # # (Prepare - hardcoded mess coming up!)
            if len(exp_runs) > 1:
                # idxs sets the correct indexes to fill in y_values array
                if 'exp_5' in str(exp_run):
                    idxs = (20*exp_idx, 19 + 20*exp_idx)
                elif 'exp_6' in str(exp_run):
                    idxs = (98, 100)
                else:
                    idxs = (20*exp_idx, 20 + 20*exp_idx)
                acq_times[idx, idxs[0]:idxs[1]] = np.array(data['acq_times'])
            else:
                acq_times[idx, :] = np.array(data['acq_times'])

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    for i in range(N):
        mean = np.mean(acq_times[i,:])
        std_dev = np.std(acq_times[i,:])
        ax.bar(i, mean, align='center', alpha=.5,
            log=True, color='blue')
        val = str(f'{round(mean,2)}') + r'$\pm$' + \
            str(f'{round(std_dev,2)}')
        if i < 2:
            ax.annotate(val, [i-0.3, mean+0.1*mean])
        elif i == 2:
            ax.annotate(val, [i-0.4, mean+0.1*mean])
        else:
            ax.annotate(val, [i-0.55, mean+0.1*mean])
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(NAMES[:N])
    ax.set_xlabel('fidelity')
    ax.set_ylabel('log. mean acq. time [s]')
    plt.title(r' Acquisition times in format $\bar{t} \pm \sigma$')
    plt.savefig(''.join(('../../results/figs/', figname)))
    plt.close()
    return acq_times


def plot_acq_times_histograms(acq_times, exp_names=NAMES,
                              figname='acq_times_histograms.pdf'):
    N = acq_times.shape[0] // 2
    if N % 2 == 0:
        N = acq_times.shape[0] // 2
        fig, axs = plt.subplots(nrows=N, ncols= N, figsize=(9, 9),
                                constrained_layout=True)
        for i in range(acq_times.shape[0]):
            ax = axs[i // 2][i % 2]
            ax.hist(acq_times[i, :], bins=50, alpha=.5, color='blue')
            ax.axvline(acq_times[i, :].mean(), color='k', linestyle='dashed',
                    linewidth=3, label='mean')
            ax.axvline(np.median(acq_times[i, :]), color='r', linestyle='dashed',
                    linewidth=3, alpha=.3, label='median')
            ax.set_title(exp_names[i])
            ax.set_xlabel(r'$t$ [s]')
            axs[N-1, N-1].legend()
    elif N % 2 == 1:
        N = acq_times.shape[0]
        fig, axs = plt.subplots(nrows=1, ncols= N, figsize=(16, 6),
                                constrained_layout=True)
        for i in range(acq_times.shape[0]):
            ax = axs[i]
            ax.hist(acq_times[i, :], bins=50, alpha=.5, color='blue')
            ax.axvline(acq_times[i, :].mean(), color='k', linestyle='dashed',
                    linewidth=3,
                    label=f'mean: {round(acq_times[i, :].mean(), 2)}')
            ax.axvline(np.median(acq_times[i, :]), color='r',
                       linestyle='dashed', linewidth=3, alpha=.3,
                       label=f'median: {round(np.median(acq_times[i, :]), 2)}')
            ax.set_title(exp_names[i])
            ax.set_xlabel(r'$t$ [s]')
            axs[i].legend()

    plt.savefig(''.join(('../../results/figs/', figname)))
    if show_plots:
        plt.show()
    plt.close()


def calculate_correlation_matrix(y_values):
    """Return Pearson correlation coefficients.

    Relationship between correlation coefficient matrix R and the
    covariance matrix C is:


    Args:
        y_values (ndarray): Array containing the energies for the sobol
        experiments.

    Returns:
        correlation_matrix (ndarray): Pearson correlation coefficients
    """
    if verbose:
        print("Correlation matrix, experiments in the order:\n", *NAMES)
        print(np.corrcoef(y_values).round(decimals=4))
    return np.corrcoef(y_values)


def calculate_summary_statistics(y_values):
    """Return max, min, var and amplitude of the y_value arrays.

    Args:
        y_values (ndarray): Array containing the energies for the sobol
        experiments.

    Returns:
        max, min, var, amplitude (ndarray): Array containing the quantities.
    """
    summary_statistics = np.zeros((y_values.shape[0], 5))
    for exp_idx in range(y_values.shape[0]):
        mean = np.mean(y_values[exp_idx, :])
        max_val = max(y_values[exp_idx, :])
        min_val = min(y_values[exp_idx, :])
        amplitude = 0.5 * (max_val - min_val)
        var = np.var(y_values[exp_idx, :])
        stats = np.array((mean, var, min_val, max_val, amplitude))
        summary_statistics[exp_idx] = stats
    if verbose:
        print("Experiments in the order:\n ", *NAMES)
        print("Mean, variance, min, max, amplitude")
        print(summary_statistics)
    return summary_statistics


if __name__ == '__main__':
    args = sys.argv
    if '-v' or '--verbose' in sys.argv:
        verbose = True
    if '--show_plots' in sys.argv:
        show_plots = True
    main()
