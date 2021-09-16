import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
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
#print(CONFIG["tolerances"])
#TOL_IDX = 3         # 5 : 0.1 kcal/mol, 3 : 0.5 kcal/mol
SCATTER_DICT = {'color': 'blue', 'alpha': .4, 'marker': 'x',
                'label': 'observation'}
MEANS_DICT = {'color': 'red', 'marker': '*', 's': 100}
TITLE_DICT = {'4D': {'4HFICM1': 'LF ➞ HF',
                     '4UHFICM1_r': 'LF ➞ UHF',
                     '4UHFICM2_r': 'HF ➞ UHF',
                     '4UHFICM3_r': 'LF ➞ UHF',
                     '4UHFICM4_r': 'HF ➞ UHF'},
              '2D': {'2HFICM1': 'LF ➞ HF',
                     '2UHFICM1': 'LF ➞ UHF',
                     '2UHFICM2': 'HF ➞ UHF'}}
PLOT_IDX_DICT = {'4D': {'4HFICM1': 0, '4UHFICM1_r': 1, '4UHFICM2_r': 2,
                 '4UHFICM3_r': 1, '4UHFICM4_r': 2},
                 '2D': {'2HFICM1': 0,
                     '2UHFICM1': 1,
                     '2UHFICM2': 2}}
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 12, 20, 25


def main(args):
    tolerance = args.tolerance
    tolerances = np.array(CONFIG['tolerances'])
    if tolerance not in tolerances:
        raise Exception(f"Invalid tolerance level, chose from {tolerances}")
    TOL_IDX = np.argwhere(args.tolerance == tolerances).squeeze()
    config = CONFIG[f'TL_experiment_plots_{args.dimension}']
    figname = args.dimension + '_tol_' + str(tolerance) + '_TL.pdf'
    tl_experiments = [THESIS_DIR / 'data' / 'processed' /
                      exp for exp in config.keys()]
    baseline_experiments = [THESIS_DIR / 'data' / 'processed' /
                            config[exp][0] for exp in config]
    tl_experiment_data = load_experiments(tl_experiments)
    baseline_experiment_data = load_experiments(baseline_experiments)
    plot_tl_convergence(figname, baseline_experiment_data, tl_experiment_data,
                        tol_idx=TOL_IDX, show_plots=args.show_plots)


def load_experiments(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        for exp in experiment.iterdir():
            if exp.is_file():
                exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def plot_tl_convergence(figname, baseline_experiments, tl_experiments,
                        tol_idx=5, show_plots=False):
    N = len(tl_experiments)
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))

    for exp_idx in range(N):
        bl_iterations, bl_times = [], []        # bl : baseline
        tl_iterations, tl_times = [], []        # tl : transfer_learning
        baseline_data = baseline_experiments[exp_idx]
        tl_data = tl_experiments[exp_idx]
        max_iterations, max_time = 0, 0         # used to scale axes
        idx = None
        for bl, tl, in zip(baseline_data, tl_data):
            name = tl['name']
            idx = PLOT_IDX_DICT[args.dimension][name]
            bl_initpts, tl_initpts = bl['initpts'][1], tl['initpts'][1]
            bl_conv, tl_conv = bl['iterations_to_gmp_convergence'][tol_idx], \
                tl['iterations_to_gmp_convergence'][tol_idx]
            bl_conv_time, tl_conv_time = bl['totaltime_to_gmp_convergence'][tol_idx], \
                tl['totaltime_to_gmp_convergence'][tol_idx]
            bl_conv_time /= 3600
            bl_iterations.append(bl_conv)
            bl_times.append(bl_conv_time)
            if exp_idx < 3:
                axs[0, idx].scatter(bl_initpts, bl_conv, **SCATTER_DICT)
                axs[1, idx].scatter(bl_initpts, bl_conv_time, **SCATTER_DICT)
            if tl_conv is None or tl_conv_time is None:
                continue
            tl_conv_time /= 3600
            tl_iterations.append(tl_conv)
            tl_times.append(tl_conv_time)
            max_iterations = max(max_iterations, bl_conv, tl_conv)
            max_time = max(max_time, bl_conv_time, tl_conv_time)
            axs[0, idx].scatter(tl_initpts, tl_conv, **SCATTER_DICT)
            axs[1, idx].scatter(tl_initpts, tl_conv_time, **SCATTER_DICT)
        if exp_idx < 3:
            axs[0, idx].scatter(bl_initpts, np.mean(bl_iterations),
                                    **MEANS_DICT)
            axs[0, idx].text(0.05*tl_initpts, 1.02*np.mean(bl_iterations),
                                f'{int(np.mean(bl_iterations))}', c='r')
            axs[1, idx].scatter(bl_initpts, np.mean(bl_times),
                                    **MEANS_DICT)
            axs[1, idx].text(0.05*tl_initpts, 1.02*np.mean(bl_times),
                                f'{round(np.mean(bl_times), 1)}', c='r')
        axs[0, idx].scatter(tl_initpts, np.mean(tl_iterations),
                                **MEANS_DICT)
        axs[0, idx].text(0.78*tl_initpts, 1.02*np.mean(tl_iterations),
                             f'{int(np.mean(tl_iterations))}', c='r')
        axs[1, idx].scatter(tl_initpts, np.mean(tl_times),
                                **MEANS_DICT)
        axs[1, idx].text(0.78*tl_initpts, 1.02*np.mean(tl_times),
                             f'{round(np.mean(tl_times), 1)}', c='r')
        axs[0, idx].set_ylim([0, max_iterations+0.1*max_iterations])
        axs[1, idx].set_ylim([0, max_time+0.1*max_time])
        axs[0, idx].set_title(f'{TITLE_DICT[args.dimension][name]}',
                    fontsize=SMALL_SIZE)
        axs[0, idx].set_xticks([])
        # axs[1, exp_idx].set_title(f'TL: {round(100, 1)} % baseline resources',
        #             fontsize=10)
    axs[0, 0].set_ylabel('BO iterations', fontsize=SMALL_SIZE)
    axs[1,0].set_ylabel('CPU time [h]', fontsize=SMALL_SIZE)
    for ax in axs[1, :]:
        ax.set_xlabel('secondary initpoints', fontsize=SMALL_SIZE)
    fig.suptitle(f'{args.dimension} TL experiments (tolerance: {args.tolerance} kcal/mol)',
                 fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    if not show_plots:
        plt.savefig(FIGS_DIR.joinpath(figname), dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--show_plots',
                        action='store_true',
                        dest='show_plots',
                        help="Show (and don't save) plots.")
    parser.add_argument('-d', '--dimension',
                        type=str,
                        help="Chose between 2D or 4D.")
    parser.add_argument('-t', '--tolerance',
                        type=float,
                        default=0.1,
                        help='Tolerance level to plot convergence for.')
    args = parser.parse_args()
    main(args)
