import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
# import pylustrator
# pylustrator.start()

import sys
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config.yaml')
TOL_IDX = 5         # 5 : 0.1 kcal/mol, 3 : 0.5 kcal/mol
if len(sys.argv) > 1:
    CONFIG = CONFIG[f'TL_experiment_plots_{sys.argv[1]}']
    figname = sys.argv[1] + '_TL.pdf'
else:
    raise Exception("Usage: python3 plot_TL_convergence.py 2D (or 4D)")

tl_experiments = [THESIS_DIR / 'data' / 'processed' /
                  exp for exp in CONFIG.keys()]
print(tl_experiments)

# baselines, corresponding to each tl experiment
baseline_experiments = [THESIS_DIR / 'data' / 'processed' /
                  CONFIG[exp][0] for exp in CONFIG]
SCATTER_DICT = {'color': 'blue', 'alpha': .4, 'marker': 'x',
                'label': 'observation'}
FIT_DICT = {'color': 'red', 'label': 'trend', 'linewidth': 3,
            'linestyle': 'dashed', 'alpha': .5}
MEANS_DICT = {'color': 'red', 'marker': '*', 's': 100}
TITLE_DICT = {0: 'LF ➞ HF', 1: 'LF ➞ UHF', 2: 'HF ➞ UHF'}
SMALL_SIZE = 12
MEDIUM_SIZE = 20
LARGE_SIZE = 25


def main():
    tl_experiment_data = load_experiments(tl_experiments)
    baseline_experiment_data = load_experiments(baseline_experiments)
    plot_tl_convergence(figname,
                        baseline_experiment_data, tl_experiment_data)


def load_experiments(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        for exp in experiment.iterdir():
            if exp.is_file():
                exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def plot_tl_convergence(figname, baseline_experiments, tl_experiments):

    N = len(tl_experiments)
    print(N)
    fig, axs = plt.subplots(2, N, figsize=(3*N, 6))
    mean_values = np.zeros((2, *axs.shape))



    for tl_exp_idx in range(N):
        baseline_data = baseline_experiments[tl_exp_idx]
        tl_data = tl_experiments[tl_exp_idx]

        explist = baseline_data
        name = explist[0]['name']
        for tl_exp in tl_data:
            explist.append(tl_exp)

        convergence_iterations, convergence_times = [], []
        for exp in explist:
            if len(exp['initpts']) > 1:
                secondary_initpts = int(exp['initpts'][1])
            else:
                secondary_initpts = 0
            convergence_iter = exp['iterations_to_gmp_convergence'][TOL_IDX]
            convergence_iterations.append([secondary_initpts,
                                           convergence_iter])

            convergence_time = exp['totaltime_to_gmp_convergence'][TOL_IDX]
            convergence_times.append([secondary_initpts,
                                      convergence_time])

        for quantity_idx, quantity in enumerate([convergence_iterations,
                                                 convergence_times]):
            # scatter
            quantity = np.array(quantity, dtype=float)
            if quantity_idx == 1:
                quantity[:, 1] /= 3600
            axs[quantity_idx, tl_exp_idx].scatter(quantity[:, 0],
                                                  quantity[:, 1],
                                                  **SCATTER_DICT)
            # fit
            # TODO : Uncomment once all runs have converged, this raises
            # error because LinearRegression doesn't work with NaN values
            x = quantity[:, 0].reshape(-1, 1)
            y = quantity[:, 1].reshape(-1, 1)
            # reg = LinearRegression().fit(x, y)
            # x_plot = np.arange(0, 50, 0.01).reshape(-1, 1)
            # y_plot = reg.predict(x_plot)
            # axs[quantity_idx, tl_exp_idx].plot(x_plot, y_plot, **FIT_DICT)
            # means
            for initpts_idx, initpts in enumerate(np.unique(x)):
                # Can be removed once all runs have converged
                if '4' in name:
                    y_without_nan = [x for x in y[x == initpts] if
                        str(x) != 'nan']
                    mean = np.mean(y_without_nan)
                else:
                    mean = np.mean(y[x == initpts])
                mean_values[initpts_idx, quantity_idx, tl_exp_idx] = mean
                if initpts_idx == 0:
                    axs[quantity_idx, tl_exp_idx].scatter(
                        [initpts], [mean], **MEANS_DICT, label='mean')
                    axs[quantity_idx, tl_exp_idx].text(10, mean, f'{int(mean)}', c='r')
                else:
                    axs[quantity_idx, tl_exp_idx].scatter(
                        [initpts], [mean], **MEANS_DICT)
            # Raises warning but can be ignored, this is just used for titles
            reduction_values = mean_values[1, :, :] / mean_values[0, :, :]
            reduction = reduction_values[quantity_idx, tl_exp_idx]
            if quantity_idx == 0:
                axs[quantity_idx, tl_exp_idx].set_title(
                    f'{TITLE_DICT[tl_exp_idx % 3]}\n' +
                    f'TL: {round(100*reduction, 1)} % baseline resources',
                    fontsize=10)
            else:
                axs[quantity_idx, tl_exp_idx].set_title(
                    f'TL: {round(100*reduction, 1)} % baseline resources',
                    fontsize=10)

            axs[0, 0].legend(fontsize=SMALL_SIZE)
    # axs[0, 0].set_xticks([])
    # axs[0, 1].set_xticks([])
    # if '4' in name:
    #     axs[1, 0].set_xticks([0, 100, 200])
    #     axs[1, 1].set_xticks([0, 100, 200])
    # else:
    #     axs[1, 0].set_xticks([0, 25, 50])
    #     axs[1, 1].set_xticks([0, 25, 50])
    axs[0, 0].set_ylabel('BO iterations',
                        fontsize=SMALL_SIZE)
    axs[1,0].set_ylabel('CPU time [h]', fontsize=SMALL_SIZE)
    for ax in axs[1, :]:
        ax.set_xlabel('secondary initpoints', fontsize=SMALL_SIZE)

    fig.suptitle(f'{name[:1]}D TL experiments',
                 fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    if '--display' in sys.argv[1:]:
        plt.show()
    else:
        plt.savefig(FIGS_DIR.joinpath(figname), dpi=300)


main()
