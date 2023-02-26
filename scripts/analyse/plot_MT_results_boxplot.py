import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import click

from pathlib import Path
from src.read_write import load_yaml, load_experiments, \
    load_statistics_to_dataframe

plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config_mt.yaml')
EXP_NAMES = ['basic', 'ELCB1', 'ELCB3', 'ELCB6', 'MES1', 'MES3', 'MES6',
             'MUMBO']

y_ticks_uhf = [0.1, 0.2, 0.5, 1.0]
y_ticks_hf = [0.2, 0.5, 1.0, 2.0, 5.0]
y_ticks_hf_bo_2D = [0.3, 0.5, 1.0]
y_ticks_hf_cpu_2D = [1.0, 1.5, 3.0]
y_ticks_hf_bo_4D = [0.2, 0.3, 0.5, 1.0]
y_ticks_hf_cpu_4D = [2.0, 3.0, 5.0]


@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--dimension', default='2D', type=str,
              help='Chose between 2D or 4D.')
@click.option('--tolerance', default=0.23, type=float,
              help='Tolerance level to plot convergence for.')
@click.option('--highest_fidelity', default='uhf', type=str,
              help="Chose between 'uhf' and 'hf'.")
@click.option('--print_non_converged', default=False, is_flag=True,
              help='Print a sub-dataframe of unconverged runs')
@click.option('--print_summary', default=False, is_flag=True,
              help='Print summary statistics of the dataframe to the terminal')
def main(show_plots, dimension, tolerance, highest_fidelity,
         print_non_converged, print_summary):
    tolerances = np.array(CONFIG['tolerances'])
    if tolerance not in tolerances:
        raise Exception(f"Invalid tolerance level, chose from {tolerances}")
    #config = CONFIG[f'TL_experiment_plots_{dimension}']
    config = CONFIG[f'MT_experiment_plots_{dimension}']

    tl_experiments = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                      exp for exp in config.keys()]

    # Don't load baseline experiments multiple times
    bl_experiment_keys = list(set([config[exp][0] for exp in config]))
    bl_experiments = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                      exp for exp in bl_experiment_keys]
    tl_exp_data = load_experiments(tl_experiments)
    bl_exp_data = load_experiments(bl_experiments)
    df = load_statistics_to_dataframe(bl_exp_data, tl_exp_data, num_exp=5)
#    df.to_csv('mt_test.csv')
    plot_convergence_as_boxplot(
        df, tolerance, dimension, highest_fidelity, show_plots,
        print_non_converged, print_summary)


def plot_convergence_as_boxplot(
        df, tolerance, dimension, highest_fidelity, show_plots,
        print_not_converged, print_summary):
    tolerance_idx = np.argwhere(
        tolerance == np.array(np.array(CONFIG['tolerances']))).squeeze()

    plot_df = df[['name', 'iterations_to_gmp_convergence',
                  'totaltime_to_gmp_convergence',
                  'highest_fidelity_iterations_to_gmp_convergence']]
    plot_df.index = range(len(plot_df))
    df_copy = plot_df.copy()
    df_copy['iterations'] = plot_df[
        'iterations_to_gmp_convergence'].map(lambda x: x[tolerance_idx])
    df_copy['BO iter.'] = plot_df[
        'highest_fidelity_iterations_to_gmp_convergence'].map(
            lambda x: x[tolerance_idx])
    df_copy['CPU t [h]'] = plot_df[
        'totaltime_to_gmp_convergence'].map(
            lambda x: chose_value_with_tolerance(x, tolerance_idx))
    yticks_bo, yticks_cpu = choose_yticks(highest_fidelity, dimension)
    if highest_fidelity == 'uhf':
        df_copy = df_copy[df_copy['name'].str.contains('UHF') == True]
    elif highest_fidelity == 'hf':
        df_copy = df_copy[df_copy['name'].str.contains('UHF') == False]
    else:
        raise Exception("Invalid highest fidelity")
    fig_width = 4.0 if highest_fidelity == "hf" else 6.5
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, 4.5))
    conditions_strategy = [
        (df_copy['name'].str.contains('basic')),
        (df_copy['name'].str.contains('ICM1')),
        (df_copy['name'].str.contains('ICM2'))]
    strategies = ['BL', r'LF $\rightarrow$ UHF', r'HF $\rightarrow$ UHF']
    if highest_fidelity == 'hf':
        strategies[1] = r'LF $\rightarrow$ HF'
    conditions_approach = [
        (df_copy['name'].str.contains(approach)) for approach in EXP_NAMES
    ]
    approaches = ['BL', 'ELCB 1', 'ELCB 3', 'ELCB 6', 'MES 1', 'MES 3',
                  'MES 6', 'MUMBO']
    if print_not_converged:
        iterations_df = df_copy[['name', 'iterations']]
        print(iterations_df[iterations_df['iterations'].isna()])
    df_copy['Setup'] = np.select(conditions_strategy, strategies)
    df_copy['Strategy'] = np.select(conditions_approach, approaches)
    if print_summary:
        print(df_copy.groupby(['Setup', 'Strategy'])['CPU t [h]'].describe(
            percentiles=[.5]).round(2))

    plot_relative_to_baseline = True
    if plot_relative_to_baseline:
        if highest_fidelity == "uhf":
            baseline_name = "4UHFbasic1_r" if dimension == '4D' else "2UHFbasic1_r"
        elif highest_fidelity == "hf":
            baseline_name = "4HFbasic1" if dimension == '4D' else "2HFbasic1"
        mean_bo_iter = int(
            df_copy[df_copy["name"] == baseline_name]["BO iter."].mean())
        mean_time = df_copy[
            df_copy["name"] == baseline_name]["CPU t [h]"].mean()
    df_copy = df_copy[~df_copy["name"].str.contains("basic")]
    df_copy["BO iter."] /= mean_bo_iter
    df_copy["CPU t [h]"] /= mean_time
    axs[0] = sns.boxplot(x='Setup', y='BO iter.', hue='Strategy',
                whis=[0.25, 0.75], data=df_copy, palette="tab10", ax=axs[0],
                showmeans=True, meanprops={"marker": "x", "markersize": 6,
                                           "markeredgecolor": "black",
                                           "markeredgewidth": 1})
    axs[0].set_yscale("log")
    axs[0].set_yticks(yticks_bo, yticks_bo)

    axs[1] = sns.boxplot(x='Setup', y='CPU t [h]', hue='Strategy', whis=[0.25, 0.75],
                data=df_copy, palette="tab10", ax=axs[1], showmeans=True,
                meanprops={"marker": "x", "markersize": 6,
                           "markeredgecolor": "black",
                           "markeredgewidth": 1})
    axs[1].set_yscale("log")
    axs[1].set_yticks(yticks_cpu, yticks_cpu)

    axs[0].set_xlabel('')
    if (dimension == '2D') and (highest_fidelity == 'hf'):
        location, legend_idx = 'upper left', 1
    elif (dimension == '4D') and (highest_fidelity == 'uhf'):
        location, legend_idx = 'upper right', 0
    else:
        location, legend_idx = 'lower left', 0
    if highest_fidelity == "uhf":
        axs[legend_idx].legend(loc=location, fontsize=10, ncol=2, columnspacing=.5)
        axs[1 - legend_idx].get_legend().remove()
    elif highest_fidelity == "hf":
        axs[0].get_legend().remove()
        axs[1].get_legend().remove()
    axs[0].set_ylabel('BO iter.', fontsize=14)
    axs[1].set_ylabel('CPU t [h]', fontsize=14)
    if plot_relative_to_baseline:
        axs[0].set_ylabel('Relative\nBO iter.', fontsize=14)
        axs[1].set_ylabel('Relative\nCPU time', fontsize=14)
    axs[1].set_xlabel('')
    axs[0].set_xticks([])
    fig.tight_layout()

    if show_plots:
        plt.show()
    else:
        name = f'MT_convergence_boxplot_{dimension}_{highest_fidelity}'
        plt.savefig(FIGS_DIR / f'{name}.pdf')


def chose_value_with_tolerance(x, tolerance_idx, time_in_h=True):
    if x[tolerance_idx] is None:
        return None
    else:
        x_tol = x[tolerance_idx] / 3600 if time_in_h else x[tolerance_idx]
        return x_tol


def choose_yticks(fidelity, dimension):
    if fidelity == "uhf":
        return y_ticks_uhf, y_ticks_uhf
    elif fidelity == "hf":
        if dimension == "2D":
            return y_ticks_hf_bo_2D, y_ticks_hf_cpu_2D
        elif dimension == "4D":
            return y_ticks_hf_bo_4D, y_ticks_hf_cpu_4D


if __name__ == '__main__':
    main()
