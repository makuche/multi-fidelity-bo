import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


SMALL_SIZE = 15
MEDIUM_SIZE = 20
LARGE_SIZE = 30

THESIS_DIR = Path(__name__).resolve().parent.parent.parent
# TODO : Take the list of sobol experiments to yaml config file
SOBOL_EXPERIMENTS = [
    '2UHFbasic0',
    '2UHF0basic0',
    '2HFbasic0',
    '2LFbasic0'
                    ]
NAMES = ['UHF', 'UHF0', 'HF', 'LF']
# Creating a list with all exp_*.json to loop over
exp_list = [
    THESIS_DIR / 'data' / 'processed' / exp  for \
    exp in SOBOL_EXPERIMENTS
           ]


def main():

    plot_y_scatter_trellis(exp_list)
    acq_times = plot_acq_times_comparison(exp_list)
    plot_acq_times_histograms(acq_times, NAMES)
    #correlation_matrix = calculate_correlation_matrix(exp_list)


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
            with open(f'{exp_run}', 'r') as file:
                data = json.load(file)
                # (Prepare - hardcoded mess coming up!)
                if len(exp_runs) > 1:
                    # idxs sets the correct indexes to fill in y_values array
                    if 'exp_5' in str(exp_run):
                        idxs = (20*exp_idx, 19 + 20*exp_idx)
                    else:
                        idxs = (20*exp_idx, 20 + 20*exp_idx)
                    y_values[idx, idxs[0]:idxs[1]] = np.array(data['xy'])[:,-1]
                    if 'exp_5' in str(exp_run):
                        # TODO : Fix this with correct truemin!
                        y_values[idx, idxs[0]:idxs[1]] -= -202857.96512
                else:
                    y_values[idx,:] = np.array(data['xy'])[:data_points,-1]

    fig, axs = plt.subplots(N,N, figsize=(5*N,5*N), constrained_layout=True)
    for i in range(N):
        ax = axs[i,i]
        ax.axis('off')
        ax.text(0.5, 0.5, f'{NAMES[i]}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=30,
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
        # remove below-diagonal plots
        for j in range(i):
            ax = axs[i,j]
            ax.axis('off')
    axs[0,N-1].set_xticklabels([])
    axs[0,N-1].set_yticklabels([])
    plt.show()
    plt.savefig(''.join(('../../results/figs/', figname)))
    plt.close()


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
        for exp_idx, exp_run in enumerate(exp_runs):
            with open(f'{exp_run}', 'r') as file:
                data = json.load(file)
                # # (Prepare - hardcoded mess coming up!)
                if len(exp_runs) > 1:
                    # idxs sets the correct indexes to fill in y_values array
                    if 'exp_5' in str(exp_run):
                        idxs = (20*exp_idx, 19 + 20*exp_idx)
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
    fig, axs = plt.subplots(nrows=N, ncols= N, figsize=(9, 9),
                            constrained_layout=True)
    for i in range(acq_times.shape[0]):
        ax = axs[i // 2][i % 2]
        ax.hist(acq_times[i, :], bins=3 0, alpha=.5, color='blue')
        ax.set_title(exp_names[i])
        ax.set_xlabel(r'$t$ [s]')
    plt.savefig(''.join(('../../results/figs/', figname)))
    plt.close()


# TODO : Continue implementing this function
def calculate_correlation_matrix(exp_list):
    N = len(exp_list)
    correlation_matrix = np.zeros((N, N))
    return
    for i in range(N):
        exp_1 = np.array(exp_list[[i]])
    return correlation_matrix


if __name__ == '__main__':
    main()
