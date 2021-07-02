import numpy as np
import matplotlib.pyplot as plt
import json

def plot_y_scatter_trellis(exp_list, figname='trellis_correlation.pdf'):
    """Create scatter trellis plot of sobol que experiments.

    Args:
        exp_list (list): List containing the experiments .json file paths
        figname (str, optional): File name for figure. Defaults to
        'linear_correlation.pdf'.
    """
    N = len(exp_list)
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    LARGE_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    y_values = []
    names = []
    for exp_path in exp_list:
        with open(f'{exp_path}', 'r') as file:
            data = json.load(file)
            y_values.append(np.array(data['xy'])[:,-1])
            names.append(data['name'])
    y_values = np.array(y_values)      # (N,initpts) shape


    fig, axs = plt.subplots(N,N, figsize=(5*N,5*N), constrained_layout=True)
    for i in range(N):
        ax = axs[i,i]
        ax.axis('off')
        ax.text(0.5, 0.5, f'{names[i]} (kcal/mol)',
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


exp_list = [
    '/home/manuel/Dropbox/Studium/Master/Thesis \
        Project/thesis/data/processed/a1a1/exp_1.json',
    '/home/manuel/Dropbox/Studium/Master/Thesis \
        Project/thesis/data/processed/a1b1/exp_1.json',
    '/home/manuel/Dropbox/Studium/Master/Thesis \
        \Project/thesis/data/processed/UHF/exp_1.json'
    ]

plot_y_scatter_trellis(exp_list)