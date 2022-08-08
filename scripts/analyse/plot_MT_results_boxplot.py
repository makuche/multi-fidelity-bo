from pathlib import Path
from src.read_write import load_yaml, load_experiments, \
    load_statistics_to_dataframe
from posixpath import split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })
import seaborn as sns
import pandas as pd
import click


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config_mt.yaml')


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

    if highest_fidelity == 'uhf':
        df_copy = df_copy[df_copy['name'].str.contains('UHF') == True]
    elif highest_fidelity == 'hf':
        df_copy = df_copy[df_copy['name'].str.contains('UHF') == False]
    else:
        raise Exception("Invalid highest fidelity")
    fig, axs = plt.subplots(2, 1, figsize=(6.5, 3))
    conditions_strategy = [
        (df_copy['name'].str.contains('basic') ),
        (df_copy['name'].str.contains('ICM1')),
        (df_copy['name'].str.contains('ICM2'))]
    strategies = ['BL', r'LF $\rightarrow$ UHF',
                  r'HF $\rightarrow$ UHF']
    if highest_fidelity == 'hf':
        strategies[1] = r'LF $\rightarrow$ HF'
    conditions_approach = [
        (df_copy['name'].str.contains('basic')),
        (df_copy['name'].str.contains('ELCB1')),
        (df_copy['name'].str.contains('ELCB3')),
        (df_copy['name'].str.contains('ELCB6'))]
    approaches = ['BL', 'MFBO 1',
                  'MFBO 3',
                  'MFBO 6']
    if print_not_converged:
        iterations_df = df_copy[['name', 'iterations']]
        print(iterations_df[iterations_df['iterations'].isna()])
    df_copy['Setup'] = np.select(conditions_strategy, strategies)
    df_copy['Strategy'] = np.select(conditions_approach, approaches)
    if print_summary:
        print(df_copy.groupby(['Setup', 'Strategy'])\
            ['CPU t [h]'].describe(percentiles=[.5]).round(2))
    sns.boxplot(x='Setup', y='BO iter.', hue='Strategy',
                whis=[0.25, 0.75], data=df_copy, palette="tab10", ax=axs[0],
                showmeans=True, meanprops={"marker": "x", "markersize": 6,
                                           "markeredgecolor": "black",
                                           "markeredgewidth": 1})
    sns.boxplot(x='Setup', y='CPU t [h]', hue='Strategy', whis=[0.25, 0.75],
                data=df_copy, palette="tab10", ax=axs[1], showmeans=True,
                meanprops={"marker": "x", "markersize": 6,
                           "markeredgecolor": "black",
                           "markeredgewidth": 1})
    axs[0].set_xlabel('')
    if (dimension == '2D') and (highest_fidelity == 'hf'):
        location, legend_idx = 'upper left', 1
    elif (dimension == '4D') and (highest_fidelity == 'uhf'):
        location, legend_idx = 'upper right', 0
    else:
        location, legend_idx = 'lower left', 0
    axs[legend_idx].legend(loc=location, fontsize=10, ncol=2, columnspacing=.5)
    axs[1 - legend_idx].get_legend().remove()
    axs[0].set_ylabel('BO iter.', fontsize=14)
    axs[1].set_ylabel('CPU t [h]', fontsize=14)
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

if __name__ == '__main__':
    main()
