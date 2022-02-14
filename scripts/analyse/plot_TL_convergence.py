import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from pathlib import Path

from src.read_write import load_yaml, load_json


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config.yaml')
#print(CONFIG["tolerances"])
#TOL_IDX = 3         # 5 : 0.1 kcal/mol, 3 : 0.5 kcal/mol
BLUE, RED = '#000082', '#FE0000'
SCATTER_DICT = {'color': BLUE, 'alpha': .4, 'marker': 'x',
                'label': 'observation', 's': 60}
MEANS_DICT = {'color': RED, 'marker': '*', 's': 120, 'label': 'mean'}
TITLE_DICT = {'4D': {'4HFICM1': 'LF ➞ HF',
                     '4UHFICM1_r': 'LF ➞ UHF',
                     '4UHFICM2_r': 'HF ➞ UHF',
                     '4UHFICM3_r': 'LF ➞ UHF',
                     '4UHFICM4_r': 'HF ➞ UHF'},
              '2D': {'2HFICM1': 'LF ➞ HF',
                     '2UHFICM1': 'LF ➞ UHF',
                     '2UHFICM2': 'HF ➞ UHF',
                     'MT_a3b7': 'MT_a3b7'}}
PLOT_IDX_DICT = {'4D': {'4HFICM1': 0, '4UHFICM1_r': 1, '4UHFICM2_r': 2,
                 '4UHFICM3_r': 1, '4UHFICM4_r': 2},
                 '2D': {'2HFICM1': 0,
                     '2UHFICM1': 1,
                     '2UHFICM2': 2,
                     'MT_a3b7': 0}}
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 12, 20, 25


def main(args):
    tolerance = args.tolerance
    tolerances = np.array(CONFIG['tolerances'])
    if tolerance not in tolerances:
        raise Exception(f"Invalid tolerance level, chose from {tolerances}")
    TOL_IDX = np.argwhere(args.tolerance == tolerances).squeeze()
    config = CONFIG[f'TL_experiment_plots_{args.dimension}']
    figname = args.dimension + '_tol_' + str(tolerance) + '_TL.png'
    tl_experiments = [THESIS_DIR / 'data' / 'processed' /
                      exp for exp in config.keys()]
    bl_experiments = [THESIS_DIR / 'data' / 'processed' /
                            config[exp][0] for exp in config]
    tl_exp_data = load_experiments(tl_experiments)
    bl_exp_data = load_experiments(bl_experiments)
    data_dict = {
        'bl': load_values_to_dict(bl_exp_data),
        'tl': load_values_to_dict(tl_exp_data)
    }
    # plot_TL_convergence(figname, data_dict, tol_idx=TOL_IDX,
    #                     show_plots=args.show_plots)
    plot_tl_convergence(figname, bl_exp_data, tl_exp_data,
                        tol_idx=TOL_IDX, show_plots=args.show_plots)
    # plot_tl_convergence_abstract(figname, bl_exp_data, tl_exp_data,
    #                     tol_idx=TOL_IDX, show_plots=args.show_plots)


def load_experiments(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        for exp in experiment.iterdir():
            if exp.is_file():
                exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def load_values_to_dict(data, tol_idx=5):
    # dict (key: bl/tl) -> dict (key: values,times) -> list -> dict (key: initpts[1])
    # Example for baseline strucutres:
    # 3 baseline runs (LF, HF, UHF) -> 30/5/5 subruns
    data_dict, values, times = {}, [], []

    for fidelity in data:
        conv_vals, conv_times = {}, {}
        for exp_run_idx, exp_run in enumerate(fidelity):
            initpts = exp_run['initpts'][1]
            if initpts not in conv_vals:
                conv_vals[initpts] = []
            conv_vals[initpts].append(
                exp_run['iterations_to_gmp_convergence'][tol_idx])
            if initpts not in conv_times:
                conv_times[initpts] = []
            conv_times[initpts].append(
                exp_run['totaltime_to_gmp_convergence'][tol_idx])
        values.append(conv_vals)
        times.append(conv_times)
    # returns lists with shape [len(30), len(5), len(5)]
    data_dict['values'] = values
    data_dict['times'] = times
    return data_dict


def plot_tl_convergence(figname, baseline_experiments, tl_experiments,
                        tol_idx=5, show_plots=False):
    N = len(tl_experiments)
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))

    linear_reg_data = {}
    for exp_idx in range(N):
        bl_iterations, bl_times = [], []        # bl : baseline
        tl_iterations, tl_times = [], []        # tl : transfer_learning
        baseline_data = baseline_experiments[exp_idx]
        tl_data = tl_experiments[exp_idx]
        max_iterations, max_time = 0, 0         # used to scale axes
        idx = None
        for data_idx, (bl, tl) in enumerate(zip(baseline_data, tl_data)):
            # I accidentally did 30 experiments for the LF->HF experiments,
            # but as it turns out, 10 runs gives the same mean statistics,
            # so it's sufficient to plot only 10 runs (also plotting
            # all 30 runs looks messy)
            if data_idx > 10:
                break
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
        axs[1, idx].scatter(tl_initpts, np.mean(tl_times), **MEANS_DICT)
        axs[1, idx].text(0.78*tl_initpts, 1.04*np.mean(tl_times),
                         f'{round(np.mean(tl_times), 2)}', c='r')
        setup = TITLE_DICT[args.dimension][name]
        if setup not in linear_reg_data:
            linear_reg_data[setup] = np.array([]).reshape(0, 2)

        tmp_bl = np.array([[bl_initpts, bl_time] for bl_time in bl_times])
        tmp_tl = np.array([[tl_initpts, tl_time] for tl_time in tl_times])
        tmp = np.vstack((tmp_bl, tmp_tl))
        linear_reg_data[setup] = np.vstack((linear_reg_data[setup], tmp))
        axs[0, idx].set_ylim([0, max_iterations+0.1*max_iterations])
        axs[1, idx].set_ylim([0, max_time+0.1*max_time])
        axs[0, idx].set_title(f'{TITLE_DICT[args.dimension][name]}',
                    fontsize=SMALL_SIZE)
        axs[0, idx].set_xticks([])
        axs[1, idx].set_title(f'TL: {round(100, 1)} % baseline resources',
                    fontsize=10)
    for setup_idx, setup in enumerate(linear_reg_data):
        linear_reg_data[setup] = np.unique(linear_reg_data[setup], axis=0)
        x, y = linear_reg_data[setup][:, 0], linear_reg_data[setup][:, 1]
        res = stats.linregress(x, y)
        axs[1, setup_idx].plot(x, res.intercept + res.slope *x,
                               color=RED, linestyle='dashed')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0, 0].legend(by_label.values(), by_label.keys(), fontsize=12,
               loc='upper right')
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


def plot_tl_convergence_abstract(figname, baseline_experiments, tl_experiments,
                                 tol_idx=5, show_plots=False):
    N = len(tl_experiments)
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))

    linear_reg_data = {}
    for exp_idx in range(N):
        bl_iterations, bl_times = [], []        # bl : baseline
        tl_iterations, tl_times = [], []        # tl : transfer_learning
        baseline_data = baseline_experiments[exp_idx]
        tl_data = tl_experiments[exp_idx]
        max_iterations, max_time = 0, 0         # used to scale axes
        idx = None
        for data_idx, (bl, tl) in enumerate(zip(baseline_data, tl_data)):
            # I accidentally did 30 experiments for the LF->HF experiments,
            # but as it turns out, 10 runs gives the same mean statistics,
            # so it's sufficient to plot only 10 runs (also plotting
            # all 30 runs looks messy)
            if data_idx > 10:
                break
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
                             f'{int(np.mean(bl_iterations))}', c=RED)
            axs[1, idx].scatter(bl_initpts, np.mean(bl_times),
                                    **MEANS_DICT)
            axs[1, idx].text(0.05*tl_initpts, 1.02*np.mean(bl_times),
                             f'{round(np.mean(bl_times), 1)}', c=RED)
        axs[0, idx].scatter(tl_initpts, np.mean(tl_iterations),
                                **MEANS_DICT)
        axs[0, idx].text(0.78*tl_initpts, 1.02*np.mean(tl_iterations),
                         f'{int(np.mean(tl_iterations))}', c=RED)
        axs[1, idx].scatter(tl_initpts, np.mean(tl_times), **MEANS_DICT)
        axs[1, idx].plot(tl_initpts, np.mean(tl_times), c=RED,
                         linestyle='dashed')
        shift_x, shift_y = 0, 0
        if round(np.mean(tl_times), 2) == 97.42:
            print("test")
            shift_x, shift_y = -0.17*tl_initpts, - 0.6*np.mean(tl_times)
        axs[1, idx].text(0.78*tl_initpts + shift_x,
                         1.25*np.mean(tl_times) + shift_y,
                         f'{round(np.mean(tl_times), 2)}', c=RED)
        setup = TITLE_DICT[args.dimension][name]
        if setup not in linear_reg_data:
            linear_reg_data[setup] = np.array([]).reshape(0, 2)

        tmp_bl = np.array([[bl_initpts, bl_time] for bl_time in bl_times])
        tmp_tl = np.array([[tl_initpts, tl_time] for tl_time in tl_times])
        tmp = np.vstack((tmp_bl, tmp_tl))
        linear_reg_data[setup] = np.vstack((linear_reg_data[setup], tmp))
        axs[0, idx].set_ylim([0, max_iterations+0.1*max_iterations])
        axs[1, idx].set_ylim([0, max_time+0.1*max_time])
        axs[0, idx].set_title(f'{TITLE_DICT[args.dimension][name]}',
                    fontsize=SMALL_SIZE)
        axs[0, idx].set_xticks([])
        #axs[1, idx].set_title(f'TL: {round(100, 1)} % baseline resources',
        #            fontsize=10)
    for setup_idx, setup in enumerate(linear_reg_data):
        linear_reg_data[setup] = np.unique(linear_reg_data[setup], axis=0)
    tmp = linear_reg_data['HF ➞ UHF']
    x = [0, 100, 200]
    tmp_means = [np.mean(np.squeeze(tmp[np.argwhere(tmp[:, 0]==i)])[:, 1])
                 for i in x]
    axs[1, 2].plot(x, tmp_means, c=RED, linestyle='dashed', zorder=0)
        #x, y = linear_reg_data[setup][:, 0], linear_reg_data[setup][:, 1]
        #res = stats.linregress(x, y)
        #axs[1, setup_idx].plot(x, res.intercept + res.slope *x,
        #                       color=RED, linestyle='dashed', zorder=5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label['CCSD(T) run'] = by_label['observation']
    del by_label['observation']
    by_label['mean '] = by_label['mean']
    del by_label['mean']
    axs[1, 2].legend(by_label.values(), by_label.keys(), fontsize=12,
               loc='upper right')
    axs[1, 2].set_ylabel('BO iterations', fontsize=SMALL_SIZE)
    axs[1, 2].set_ylabel('Resources in CPU time [h]', fontsize=SMALL_SIZE)
    for tuple_ in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]:
        axs[tuple_].remove()
    for ax in axs[1, :]:
        ax.set_xlabel('Number of DFT Samples', fontsize=SMALL_SIZE)
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
                        default='2D',
                        help="Chose between 2D or 4D.")
    parser.add_argument('-t', '--tolerance',
                        type=float,
                        default=0.1,
                        help='Tolerance level to plot convergence for.')
    args = parser.parse_args()
    main(args)
