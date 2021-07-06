import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


SMALL_SIZE = 15
MEDIUM_SIZE = 20
LARGE_SIZE = 30
# TODO : Use pathlib and add UHF B1 and B2
THESIS_DIR = Path(__name__).resolve().parent.parent.parent
# TODO : Take the list of sobol experiments to yaml config file
SOBOL_EXPERIMENTS = ['a1a1', 'a1b1', 'UHF_B1_sobol',
    #'UHF_B2_manual_sobol',
]
# TODO : Create a list with all exp_*.json...
exp_list = [
    THESIS_DIR / 'data' / 'processed' / exp  for \
    exp in SOBOL_EXPERIMENTS
]
N_DATA_POINTS = 40

def main():
    plot_y_scatter_trellis(exp_list, data_points=100)
    plot_acquisition_times(exp_list, data_points=100)


def plot_y_scatter_trellis(exp_list, figname='trellis_correlation.pdf', \
    data_points=N_DATA_POINTS):
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
    names = []
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir()]
        for exp_run in exp_runs:
            with open(f'{exp_run}', 'r') as file:
                data = json.load(file)
                # (Prepare - hardcoded mess coming up!)
                if 'B2' in data['name']:
                    if 'exp_1' in str(exp_run):
                        y_values[idx,:20] = np.array(data['xy'])[:,-1]
                        y_values[idx,:20] += 202863.13083
                        names.append(data['name'][:6])
                    elif 'exp_2' in str(exp_run):
                        y_values[idx,20:] = np.array(data['xy'])[:,-1]
                        y_values[idx,20:] += 202863.13083
                else:
                    y_values[idx,:] = np.array(data['xy'])[:data_points,-1]
                    names.append(data['name'][:6])


    fig, axs = plt.subplots(N,N, figsize=(5*N,5*N), constrained_layout=True)
    for i in range(N):
        ax = axs[i,i]
        ax.axis('off')
        ax.text(0.5, 0.5, f'{names[i]}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=30,
            transform=ax.transAxes)
        for j in range(i+1,N):
            ax = axs[i,j]
            axs[i,j].scatter(y_values[i,:], y_values[j,:], marker='x', \
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
    plt.savefig(''.join(('../../results/figs/', figname)))


def plot_acquisition_times(exp_list, figname='acquisition_times.pdf', \
    data_points=N_DATA_POINTS):
    exp_list = exp_list[:-1]

    N = len(exp_list)

    if N == 4:
        font = {'size': 16}
    else:
        font = {'size': 20}
    plt.rc('font', **font)

    #exp_list = exp_list[:-1]
    acq_times = np.zeros((N, data_points))
    names = ['LF', 'HF', 'UHF_B1', 'UHF_B2']
    for idx, exp_path in enumerate(exp_list):
        exp_runs = [exp for exp in exp_path.iterdir()]
        for exp_run in exp_runs:
            with open(f'{exp_run}', 'r') as file:
                data = json.load(file)
                # (Prepare - hardcoded mess coming up!)
                if 'B2' in data['name']:
                    if 'exp_1' in str(exp_run):
                        acq_times[idx,:20] = np.array(data['acq_times'])
                        names.append(data['name'][:6])
                    elif 'exp_2' in str(exp_run):
                        acq_times[idx,20:] = np.array(data['acq_times'])
                else:
                    acq_times[idx,:] = np.array(data['acq_times'])[:data_points]
                    names.append(data['name'][:6])



    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    # ax.bar(np.arange(N), means, yerr=variances, align='center', alpha=.5, \
    #        ecolor='black', capsize=10, log=True)
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
    ax.set_xticklabels(names[:N])
    ax.set_xlabel('fidelity')
    ax.set_ylabel('log. mean acq. time [s]')
    plt.title(r' Acquisition times in format $\bar{t} \pm \sigma$')
    plt.savefig(''.join(('../../results/figs/', figname)))

if __name__ == '__main__':
    main()
