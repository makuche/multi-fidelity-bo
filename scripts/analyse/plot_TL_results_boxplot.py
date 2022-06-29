from posixpath import split
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.size": 12})  # Tex rendering
import seaborn as sns
import pandas as pd
import click

from scipy import stats
from pathlib import Path

from src.read_write import load_yaml, load_json


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config.yaml')

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
                     '2UHFICM2': 2}}

sub_dataframes_2D = [('2HFbasic1', '2HFICM1'),
                     ('2UHFbasic1_r', '2UHFICM1'),
                     ('2UHFbasic1_r', '2UHFICM2')]
sub_dataframes_4D = [('4HFbasic1', '4HFICM1'),
                     ('4UHFbasic1_r', '4UHFICM1_r', '4UHFICM3_r'),
                     ('4UHFbasic1_r', '4UHFICM2_r', '4UHFICM4_r')]
titles = [r'LF $\rightarrow$ HF', r'LF $\rightarrow$ UHF',
             r'HF $\rightarrow$ UHF']

@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--dimension', default='2D', type=str,
              help='Chose between 2D or 4D.')
@click.option('--tolerance', default=0.23, type=float,
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
    df = load_statistics_to_dataframe(bl_exp_data, tl_exp_data, num_exp=10)
    df.to_csv('test.csv')
    plot_convergence_as_boxplot(df, tolerance, dimension, show_plots)


def load_experiments(experiments):
    """Given a list of experiment paths, load the data and return a list of
    the loaded experiments.

    Parameters
    ----------
    experiments : list
        List of Path objects to experiments

    Returns
    -------
    list
        Loaded experiments.
    """
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        exp_runs = [exp for exp in experiment.iterdir() if exp.is_file()]
        # The following sorts the experiments by the experiment number
        # (e.g. exp_1, exp_2, ...)
        exp_runs.sort(
            key=lambda string: int(str(string).split('_')[-1].split('.')[0]))
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


def plot_convergence_as_boxplot(df, tolerance, dimension, show_plots):
    tolerance_idx = np.argwhere(
        tolerance == np.array(np.array(CONFIG['tolerances']))).squeeze()
    plot_df = df[['name', 'iterations_to_gmp_convergence',
                  'totaltime_to_gmp_convergence', 'initpts']]
    plot_df['iterations'] = plot_df[
        'iterations_to_gmp_convergence'].map(lambda x: x[tolerance_idx])
    plot_df['totaltime'] = plot_df[
        'totaltime_to_gmp_convergence'].map(lambda x: x[tolerance_idx]/3600)
    plot_df['tl_initpts'] = plot_df[
        'initpts'].map(lambda x: x[1])
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 5), sharex=True)
    sub_dataframes = sub_dataframes_2D if dimension == '2D' \
        else sub_dataframes_4D
    for ax_idx, exp_names in enumerate(sub_dataframes):
        names = [name for name in exp_names]
        print(exp_names)
        if len(names) == 3:
            experiment_df = plot_df[
                (plot_df['name'] == names[0]) | (plot_df['name'] == names[1]) |
                (plot_df['name'] == names[2])]
        else:
            experiment_df = plot_df[
                (plot_df['name'] == names[0]) | (plot_df['name'] == names[1])]
        sns.boxplot(x='tl_initpts', y='iterations', data=experiment_df,
                    ax=axs[0, ax_idx], width=0.75, palette="Set2")
        sns.boxplot(x='tl_initpts', y='totaltime', data=experiment_df,
                    ax=axs[1, ax_idx], width=0.75, palette="Set2")
        sns.stripplot(x='tl_initpts', y='iterations', data=experiment_df,
                    ax=axs[0, ax_idx], dodge=True, marker='o', size=4,
                    color='black', jitter=True, alpha=.5)
        sns.stripplot(x='tl_initpts', y='totaltime', data=experiment_df,
                    ax=axs[1, ax_idx], dodge=True, marker='o', size=4,
                    color='black', jitter=True, alpha=.5)

    for idx in range(3):
        axs[0, idx].set_xlabel('')
        axs[0, idx].set_ylabel('')
        axs[1, idx].set_xlabel('Number of lower \nfidelity samples')
        axs[1, idx].set_ylabel('')
        if dimension == '2D':
            axs[0, idx].set_ylim(0, 35)
            axs[0, idx].set_title(titles[idx])
        else:
            axs[0, idx].set_ylim(0, 200)
            axs[0, idx].set_title(titles[idx])
    if dimension == '2D':
        axs[1, 0].set_ylim(0, 0.35)
        axs[1, 1].set_ylim(0, 100)
        axs[1, 2].set_ylim(0, 100)
    elif dimension == '4D':
        axs[0, 1].set_ylim(0, 120)
        axs[0, 2].set_ylim(0, 120)
        axs[1, 0].set_ylim(0, 2.5)
        axs[1, 1].set_ylim(0, 350)
        axs[1, 2].set_ylim(0, 350)
    axs[0, 0].set_ylabel('BO Iterations')
    axs[1, 0].set_ylabel('CPU time [h]')
    fig.suptitle(
        f'{dimension} Transfer Learning convergence results', fontsize=16)
    fig.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.savefig(f'results/figs/transfer_learning_boxplots_{dimension}.pdf')


if __name__ == '__main__':
    main()
