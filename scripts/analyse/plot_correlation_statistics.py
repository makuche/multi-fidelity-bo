import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })
import sys
import click
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.read_write import load_yaml, load_json, save_json

SMALL_SIZE = 12
MEDIUM_SIZE = 12
LARGE_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs/'
CONFIG = load_yaml(THESIS_DIR.joinpath('scripts'), '/config_tl.yaml')
# cyan: #3EE1D1, orange: #FF8C00
# blue: #000082, red: #FE0000
SCATTER_DICT_2D = {'marker': 'x', 'color': '#000082', 'alpha': 1, 's': 10}
SCATTER_DICT_4D = {'marker': 'x', 'color': '#FE0000', 'alpha': 1, 's': 10}
NAMES = ['LF', 'HF', 'UHF']
COLORS = ['blue', 'orange']

@click.command()
@click.option('--show_plots', is_flag=True, default=False)
def main(show_plots):
    exp_list_2D = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                   exp for exp in CONFIG['correlation_data']['2D']]
    exp_list_4D = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                   exp for exp in CONFIG['correlation_data']['4D']]
    y_values_2D, acq_times = load_2D_y_values_and_acq_times(exp_list_2D)
    y_values_4D = load_4D_y_values(exp_list_4D)
    plot_acq_times_comparison(acq_times, show_plots=show_plots)
    # plot_acq_times_histograms(acq_times, NAMES, show_plots=args.show_plots)
    for y_values in [y_values_2D, y_values_4D]:
        print_correlation_matrix(y_values)
    #    print_and_plot_summary_statistics(y_values,
    #                                      show_plots=args.show_plots)
    plot_correlation([y_values_2D, y_values_4D], show_plots=show_plots)
    # plot_correlation_coefficient([y_values_2D, y_values_4D], show_plots)


def load_2D_y_values_and_acq_times(exp_list, num_points=100):
    """This amount of data wrangling deserves it's own function.

    Args:
        exp_list (list): List containing paths to experiments.
        num_points (int, optional): Number of data points. Defaults to 100.

    Returns:
        (tuple): Returns y_values and acq_times
    """
    N = len(exp_list)
    y_values, acq_times = np.zeros((N, num_points)), np.zeros((N, num_points))
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir() if exp.is_file()]
        exp_runs.sort()
        for exp_idx, exp_run in enumerate(exp_runs):
            data = load_json(exp_run, '')
            if len(exp_runs) > 1:
                # idxs sets the correct indexes to fill in y_values array
                if 'exp_5' in str(exp_run):
                    idxs = (20*exp_idx, 19 + 20*exp_idx)
                elif 'exp_6' in str(exp_run):
                    idxs = (98, 100)
                else:
                    idxs = (20*exp_idx, 20 + 20*exp_idx)
                y_values[idx, idxs[0]:idxs[1]] = np.array(data['xy'])[:, -1]
                acq_times[idx, idxs[0]:idxs[1]] = np.array(data['acq_times'])
            else:
                y_values[idx, :] = np.array(data['xy'])[:num_points, -1]
                acq_times[idx, :] = np.array(data['acq_times'])
    return y_values, acq_times


def load_4D_y_values(exp_list, num_points=200):
    """This amount of data wrangling deserves it's own function.

    Args:
        exp_list (list): List containing paths to experiments.
        num_points (int, optional): Number of data points. Defaults to 200.

    Returns:
        (tuple): Returns y_values and acq_times
    """
    N = len(exp_list)
    # y_values, acq_times = np.zeros((N, num_points)), np.zeros((N, num_points))
    y_values = np.zeros((N, num_points))
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir() if exp.is_file()]
        exp_runs.sort()
        data = load_json(exp_runs[0], '')
        y_values[idx, :] = np.array(data['xy'])[:num_points, -1]
        # acq_times[idx, :] = np.array(data['acq_times'][:num_points])
    return y_values    # return y_values, acq_times


def plot_correlation(y_values, figname='correlation.pdf',
                     show_plots=False):
    N = y_values[0].shape[0]
    fig, axs = plt.subplots(1, N, figsize=(6.5, 2.5), constrained_layout=True)
    for values_idx, values in enumerate(y_values):
        i = values_idx
        SCATTER_STYLE = SCATTER_DICT_2D if values_idx == 0 else SCATTER_DICT_4D
        label = '2D' if values_idx == 0 else '4D'
        axs[0].scatter(values[0, :], values[1, :],
                       **SCATTER_STYLE, zorder=1-i, label=label)
        axs[1].scatter(values[0, :], values[2, :],
                       **SCATTER_STYLE, zorder=1-i)
        axs[2].scatter(values[1, :], values[2, :],
                       **SCATTER_STYLE, zorder=1-i)
    axs[0].legend(fontsize=10)
    axs[0].set_xlabel('LF')
    axs[0].set_ylabel('HF')
    axs[1].set_xlabel('LF')
    axs[1].set_ylabel('UHF')
    axs[2].set_xlabel('HF')
    axs[2].set_ylabel('UHF')
    for i in range(3):
        axs[i].set_xticks([0, 10, 20, 30])
        axs[i].set_yticks([0, 10, 20, 30])
        axs[i].set_xlim(-2, 35)
        axs[i].set_ylim(-2, 35)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
    if show_plots:
        plt.show()
    else:
        plt.savefig(FIGS_DIR / figname, dpi=300)
    plt.close()


def plot_acq_times_comparison(acq_times, figname='acquisition_times.pdf',
                              num_points=100, show_plots=False):

    N = acq_times.shape[0]

    if N == 4:
        font = {'size': 12}
    else:
        font = {'size': 12}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(3.25, 2.0), constrained_layout=True)
    for i in range(N):
        mean = np.mean(acq_times[i,:])/60
        std_dev = np.std(acq_times[i,:])/60
        ax.bar(i, mean, align='center', alpha=.7,
            log=True, color='#000082')
        val = str(f'{round(mean,2)}') + r'$\pm$' + \
            str(f'{round(std_dev,2)}')
        print(i, val)
        if i < 2:
            ax.annotate(val, [i-0.35, mean+0.15*mean])
        elif i == 2:
            ax.annotate(val, [i-0.65, mean+0.15*mean])
        else:
            ax.annotate(val, [i-0.55, mean+0.15*mean])
    ax.set_ylim(0, 4*10**2)
    ax.set_xticks(np.arange(N))
    NAMES = ['LF', 'HF', 'UHF']
    ax.set_xticklabels(NAMES[:N], fontsize=12)
    # ax.set_xlabel('fidelity')
    ax.set_ylabel('CPU t [min]', fontsize=14)
#    ax.set_xlabel('', fontsize=14)
#    ax.set_ylim(1, 3e4)
#    plt.title(r'Acquisition times for different simulators', fontsize=16)
    if not show_plots:
        plt.savefig(FIGS_DIR / figname, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_acq_times_histograms(acq_times, exp_names=NAMES,
                              figname='acq_times_histograms.pdf',
                              show_plots=False):
    N = acq_times.shape[0] // 2
    if N % 2 == 0:
        N = acq_times.shape[0] // 2
        fig, axs = plt.subplots(nrows=N, ncols=N, figsize=(9, 9),
                                constrained_layout=True)
        for i in range(acq_times.shape[0]):
            ax = axs[i // 2][i % 2]
            ax.hist(acq_times[i, :], bins=50, alpha=.7, color='#000082')
            ax.axvline(acq_times[i, :].mean(), color='k', linestyle='dashed',
                       linewidth=3, label='mean')
            ax.axvline(np.median(acq_times[i, :]), color='#FE0000',
                       linestyle='dashed', linewidth=3, label='median')
            ax.set_title(exp_names[i])
            ax.set_xlabel(r'$t$ [s]')
            axs[N-1, N-1].legend()
    elif N % 2 == 1:
        N = acq_times.shape[0]
        fig, axs = plt.subplots(nrows=1, ncols= N, figsize=(16, 6),
                                constrained_layout=True)
        for i in range(acq_times.shape[0]):
            ax = axs[i]
            ax.hist(acq_times[i, 1:], bins=50, alpha=.7, color='#000082')
            ax.axvline(acq_times[i, 1:].mean(), color='k', linestyle='dashed',
                       linewidth=3,
                       label=f'mean: {round(acq_times[i, 1:].mean(), 2)}')
            ax.axvline(np.median(acq_times[i, 1:]), color='#FE0000',
                       linestyle='dashed', linewidth=3,
                       label=f'median: {round(np.median(acq_times[i, 1:]), 2)}')
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
    # if show_plots:
    #     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    #     axs.boxplot(y_values.T, vert=True, patch_artist=True, labels=NAMES)
    #     axs.set_ylabel("Observed values")
    #     plt.show()


def plot_correlation_coefficient(y_values, show_plots=False):
    titles = ('Cross-covariance', 'Correlation coefficient')
    functions = (np.cov, np.corrcoef)
    for title, function in zip(titles, functions):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        for val_idx, values in enumerate(y_values):
            iterations = values.shape[1]
            correlations = np.array([function(values[:, :iter_]).round(decimals=3)
                            for iter_ in range(3, iterations)])
            axs[val_idx].plot(np.arange(len(correlations)), correlations[:, 0, 1],
                    label='B(LF,HF)')
            axs[val_idx].plot(np.arange(len(correlations)), correlations[:, 0, 2],
                    label='B(LF,UHF)')
            axs[val_idx].plot(np.arange(len(correlations)), correlations[:, 1, 2],
                    label='B(HF,UHF)')
            axs[val_idx].set_xlabel('iteration')
            axs[0].set_ylabel(title)
            axs[val_idx].legend()
        axs[0].set_title('2D', fontsize=12)
        axs[1].set_title('4D', fontsize=12)
        fig.suptitle(
            f'{title} between the fidelities over iteration', fontsize=18)
        if not show_plots:
            plt.savefig(FIGS_DIR /  f'{title}.pdf', dpi=300)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()
