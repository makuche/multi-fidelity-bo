from pathlib import Path
from src.read_write import load_yaml, load_experiments, \
    load_statistics_to_dataframe
from posixpath import split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
plt.rcParams.update({"text.usetex": True, "font.size": 12})  # Tex rendering
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
def main(show_plots, dimension, tolerance):
    tolerances = np.array(CONFIG['tolerances'])
    if tolerance not in tolerances:
        raise Exception(f"Invalid tolerance level, chose from {tolerances}")
    #config = CONFIG[f'TL_experiment_plots_{dimension}']
    config = CONFIG[f'MT_experiment_plots_{dimension}']

    tl_experiments = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                      exp for exp in config.keys()]

    bl_experiments = [THESIS_DIR / 'data/multi_task_learning' / 'processed' /
                      config[exp][0] for exp in config]
    tl_exp_data = load_experiments(tl_experiments)
    bl_exp_data = load_experiments(bl_experiments)
    df = load_statistics_to_dataframe(bl_exp_data, tl_exp_data, num_exp=5)
#    df.to_csv('mt_test.csv')
    plot_convergence_as_boxplot(df, tolerance, dimension, show_plots)


def plot_convergence_as_boxplot(df, tolerance, dimension, show_plots):
    tolerance_idx = np.argwhere(
        tolerance == np.array(np.array(CONFIG['tolerances']))).squeeze()
    plot_df = df[['name', 'iterations_to_gmp_convergence',
                  'totaltime_to_gmp_convergence']]
    plot_df['iterations'] = plot_df[
        'iterations_to_gmp_convergence'].map(lambda x: x[tolerance_idx])
    def scale_time(x, tolerance_idx=tolerance_idx):
        if x[tolerance_idx] is None:
            return None
        else:
            return x[tolerance_idx] / 3600
    plot_df['CPU time [h]'] = plot_df[
        'totaltime_to_gmp_convergence'].map(lambda x: scale_time(x))
    # plot_df['totaltime'] = plot_df[
    #     'totaltime_to_gmp_convergence'].map(lambda x: x[tolerance_idx]/3600)

    fig = plt.figure(figsize=(6.5, 5))
    plot_df_uhf = plot_df[plot_df['name'].str.contains('UHF') == False]
    #plot_df_uhf = plot_df[plot_df['name'].str.contains('UHF') == True]
    import IPython; IPython.embed()
    exit()
    conditions_strategy = [
        (plot_df_uhf['name'].str.contains('basic') ),
        (plot_df_uhf['name'].str.contains('ICM1')),
        (plot_df_uhf['name'].str.contains('ICM2'))]
    strategies = ['Baseline', 'LF support', 'HF support']
    conditions_approach = [
        (plot_df_uhf['name'].str.contains('_r')),
        (plot_df_uhf['name'].str.contains('ELCB1')),
        (plot_df_uhf['name'].str.contains('ELCB3')),
        (plot_df_uhf['name'].str.contains('ELCB6'))]
    approaches = ['Single fidelity', 'Approach 1', 'Approach 3', 'Approach 6']
    plot_df_uhf['Setup'] = np.select(conditions_strategy, strategies)
    plot_df_uhf['Strategy'] = np.select(conditions_approach, approaches)
    #import IPython; IPython.embed()
    sns.boxplot(x='Setup', y='iterations', hue='Strategy',
                data=plot_df_uhf, palette="Set2")
    if show_plots:
        plt.show()
    else:
        plt.savefig(FIGS_DIR / f'MT_convergence_boxplot_{dimension}_hf.pdf')

if __name__ == '__main__':
    main()
