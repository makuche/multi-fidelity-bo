import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from read_write import load_yaml, load_json, save_json

SMALL_SIZE = 12
MEDIUM_SIZE = 20
LARGE_SIZE = 30

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs/'
CONFIG = load_yaml(THESIS_DIR.joinpath('scripts'), '/config.yaml')

NAMES = ['LF', 'HF', 'UHF']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--show_plots',
                        action='store_true',
                        dest='show_plots',
                        help="Show (and don't save) plots")
    args = parser.parse_args()
    exp_list = [THESIS_DIR / 'data' / 'processed' /
                exp for exp in CONFIG['correlation_data']['2D']]
    y_values = plot_y_scatter_trellis(exp_list, show_plots=args.show_plots)
    acq_times = plot_acq_times_comparison(exp_list, show_plots=args.show_plots)
    plot_acq_times_histograms(acq_times, NAMES, show_plots=args.show_plots)
    print_correlation_matrix(y_values)
    print_and_plot_summary_statistics(y_values, show_plots=args.show_plots)


def plot_y_scatter_trellis(exp_list, figname='trellis_correlation.pdf',
                           data_points=100, show_plots=False):
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
        exp_runs = [exp for exp in exp_path.iterdir() if exp.is_file()]
        exp_runs.sort()
        for exp_idx, exp_run in enumerate(exp_runs):
            data = load_json(exp_run, '')
            if exp_idx == 0: print(data['name'])
            if len(exp_runs) > 1:
                # idxs sets the correct indexes to fill in y_values array
                if 'exp_5' in str(exp_run):
                    idxs = (20*exp_idx, 19 + 20*exp_idx)
                elif 'exp_6' in str(exp_run):
                    idxs = (98, 100)
                else:
                    idxs = (20*exp_idx, 20 + 20*exp_idx)
                y_values[idx, idxs[0]:idxs[1]] = np.array(data['xy'])[:, -1]
            else:
                y_values[idx, :] = np.array(data['xy'])[:data_points, -1]

    fig, axs = plt.subplots(N,N, figsize=(3*N,3*N), constrained_layout=True)
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
    if not show_plots:
        plt.savefig(FIGS_DIR / figname, dpi=300)
    else:
        plt.show()
    plt.close()
    return y_values


def plot_acq_times_comparison(exp_list, figname='acquisition_times.pdf',
                              num_points=100, show_plots=False):

    N = len(exp_list)

    if N == 4:
        font = {'size': 16}
    else:
        font = {'size': 20}
    plt.rc('font', **font)

    acq_times = np.zeros((N, num_points))
    for idx, exp_path in enumerate(exp_list[::-1]):
        exp_runs = [exp for exp in exp_path.iterdir()]
        exp_runs.sort()
        for exp_idx, exp_run in enumerate(exp_runs):
            data = load_json(exp_run, '')
            if exp_idx == 0: print(data['name'])
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
            ax.annotate(val, [i-0.55, mean+0.1*mean])
        else:
            ax.annotate(val, [i-0.55, mean+0.1*mean])
    ax.set_xticks(np.arange(N))
    names = NAMES[::-1]
    ax.set_xticklabels(names[:N])
    # ax.set_xlabel('fidelity')
    ax.set_ylabel('mean acq. time [s]')
    plt.title(r' Acquisition times in format $\bar{t} \pm \sigma$')
    if not show_plots:
        plt.savefig(FIGS_DIR / figname, dpi=300)
    else:
        plt.show()
    plt.close()
    return acq_times


def plot_acq_times_histograms(acq_times, exp_names=NAMES,
                              figname='acq_times_histograms.pdf',
                              show_plots=False):
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

    if not show_plots:
        plt.savefig(FIGS_DIR / figname, dpi=300)
    else:
        plt.show()
    plt.close()


def print_correlation_matrix(y_values):
    """Return Pearson correlation coefficients.

    Relationship between correlation coefficient matrix R and the
    covariance matrix C is:


    Args:
        y_values (ndarray): Array containing the energies for the sobol
        experiments.

    Returns:
        correlation_matrix (ndarray): Pearson correlation coefficients
    """
    print("Correlation matrix, experiments in the order:\n", *NAMES)
    print(np.corrcoef(y_values).round(decimals=3))
    print("Covariance matrix, experiments in the order:\n", *NAMES)
    print(np.cov(y_values).round(decimals=3))


def print_and_plot_summary_statistics(y_values, show_plots=False):
    """Return max, min, var and amplitude of the y_value arrays.

    Args:
        y_values (ndarray): Array containing the energies for the sobol
        experiments.

    Returns:
        max, min, var, amplitude (ndarray): Array containing the quantities.
    """
    summary_statistics = np.zeros((y_values.shape[0], 7))
    for exp_idx in range(y_values.shape[0]):
        mean = np.mean(y_values[exp_idx, :])
        max_ = max(y_values[exp_idx, :])
        min_ = min(y_values[exp_idx, :])
        amplitude = 0.5 * (max_ - min_)
        var = np.var(y_values[exp_idx, :])
        perc_25 = np.percentile(y_values[exp_idx, :], 25)
        perc_75 = np.percentile(y_values[exp_idx, :], 75)
        stats = np.array((mean, var, min_, max_, amplitude, perc_25, perc_75))
        summary_statistics[exp_idx] = stats
    print("Mean, variance, min, max, amplitude, perc_25, perc_75")
    print(summary_statistics.round(2))
    if show_plots:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        axs.boxplot(y_values.T, vert=True, patch_artist=True, labels=NAMES)
        axs.set_ylabel("Observed values")
        plt.show()


if __name__ == '__main__':
    main()
