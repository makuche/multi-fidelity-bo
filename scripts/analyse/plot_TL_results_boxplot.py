import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import click

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

@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--dimension', default='2D', type=str,
              help='Chose between 2D or 4D.')
@click.option('--tolerance', default=0.1, type=float,
              help='Tolerance level to plot convergence for.')


def main(show_plots, dimension, tolerance):
    tolerances = np.array(CONFIG['tolerances'])
    if tolerance not in tolerances:
        raise Exception(f"Invalid tolerance level, chose from {tolerances}")
    TOL_IDX = np.argwhere(tolerance == tolerances).squeeze()
    config = CONFIG[f'TL_experiment_plots_{dimension}']
    figname = dimension + '_tol_' + str(tolerance) + '_TL.png'
    tl_experiments = [THESIS_DIR / 'data/transfer_learning' / 'processed' /
                      exp for exp in config.keys()]
    bl_experiments = [THESIS_DIR / 'data/transfer_learning' / 'processed' /
                            config[exp][0] for exp in config]
    tl_exp_data = load_experiments(tl_experiments)
    bl_exp_data = load_experiments(bl_experiments)
    df = load_statistics_to_dataframe(bl_exp_data, tl_exp_data)


def load_experiments(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        exp_runs = [exp for exp in experiment.iterdir() if exp.is_file()]
        exp_runs.sort(key = int)
        print(exp_runs)
        for exp in exp_runs:
            exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def load_statistics_to_dataframe(baseline_experiments, tl_experiments,
                                 num_exp=None):
    columns = [key for key in tl_experiments[0][0].keys()]
    df = pd.DataFrame(columns=columns)


    for baseline_experiment in baseline_experiments:
        for baseline_run in baseline_experiment[:num_exp]:
            entry = correct_type_for_dataframe(baseline_run)
            df = pd.concat([df, entry], axis=0)
    for tl_experiment in tl_experiments:
        for tl_run in tl_experiment[:num_exp]:
            entry = correct_type_for_dataframe(tl_run)
            df = pd.concat([df, entry], axis=0)
    return df


def correct_type_for_dataframe(result):
    """Utility function to format results (saved in a dictionary) to
    a dataframe.

    This causes problems, if the entries in the dict have different lengthts.
    A quick fix solution, is to put all entries into a list with the
    same length.

    Parameters
    ----------
    results : dict
        Dictionary containing parsed results.

    Returns
    -------
    Dataframe
        Returns dataframe that can be concatenated to a dataframe.
    """
    entries_to_lists, lists_with_length_one = {}, {}
    for key in result:
        if not isinstance(result[key], list):
            entries_to_lists[key] = [result[key]]
        else:
            entries_to_lists[key] = result[key]
    for key in entries_to_lists:
        if len(entries_to_lists[key]) == 1:
            lists_with_length_one[key] = entries_to_lists[key]
        else:
            lists_with_length_one[key] = [entries_to_lists[key]]
    return pd.DataFrame(lists_with_length_one)


def plot_tl_convergence(figname, baseline_experiments, tl_experiments,
                        dimension, tolerance, tol_idx=5, show_plots=False):
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
            idx = PLOT_IDX_DICT[dimension][name]
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
        setup = TITLE_DICT[dimension][name]
        if setup not in linear_reg_data:
            linear_reg_data[setup] = np.array([]).reshape(0, 2)

        tmp_bl = np.array([[bl_initpts, bl_time] for bl_time in bl_times])
        tmp_tl = np.array([[tl_initpts, tl_time] for tl_time in tl_times])
        tmp = np.vstack((tmp_bl, tmp_tl))
        linear_reg_data[setup] = np.vstack((linear_reg_data[setup], tmp))
        axs[0, idx].set_ylim([0, max_iterations+0.1*max_iterations])
        axs[1, idx].set_ylim([0, max_time+0.1*max_time])
        axs[0, idx].set_title(f'{TITLE_DICT[dimension][name]}',
                    fontsize=SMALL_SIZE)
        axs[0, idx].set_xticks([])
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
    fig.suptitle(f'{dimension} TL experiments (tolerance: {tolerance} kcal/mol)',
                 fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    if not show_plots:
        plt.savefig(FIGS_DIR.joinpath(figname), dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
