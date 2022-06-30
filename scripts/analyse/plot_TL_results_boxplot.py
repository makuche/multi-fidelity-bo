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
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config_tl.yaml')


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
    config = CONFIG[f'TL_experiment_plots_{dimension}']
    tl_experiments = [THESIS_DIR / 'data/transfer_learning' / 'processed' /
                      exp for exp in config.keys()]
    bl_experiments = [THESIS_DIR / 'data/transfer_learning' / 'processed' /
                      config[exp][0] for exp in config]
    tl_exp_data = load_experiments(tl_experiments)
    bl_exp_data = load_experiments(bl_experiments)
    df = load_statistics_to_dataframe(bl_exp_data, tl_exp_data, num_exp=5)
    df.to_csv('test.csv')
    plot_convergence_as_boxplot(df, tolerance, dimension, show_plots)


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
        # sns.stripplot(x='tl_initpts', y='iterations', data=experiment_df,
        #             ax=axs[0, ax_idx], dodge=True, marker='o', size=4,
        #             color='black', jitter=True, alpha=.5)
        # sns.stripplot(x='tl_initpts', y='totaltime', data=experiment_df,
        #             ax=axs[1, ax_idx], dodge=True, marker='o', size=4,
        #             color='black', jitter=True, alpha=.5)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    for idx, labels in enumerate([('a)', 'b)'), ('c)', 'd)'), ('e)', 'f)')]):
        axs[0, idx].text(0.0, 1.0, labels[0],
                         transform=axs[0, idx].transAxes + trans, fontsize=12,
                         verticalalignment='top')
        axs[1, idx].text(0.0, 1.0, labels[1],
                         transform=axs[1, idx].transAxes + trans, fontsize=12,
                         verticalalignment='top')
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
        axs[1, 1].set_ylim(0, 120)
        axs[1, 2].set_ylim(0, 120)
    elif dimension == '4D':
        axs[0, 1].set_ylim(0, 130)
        axs[0, 2].set_ylim(0, 130)
        axs[1, 0].set_ylim(0, 2.5)
        axs[1, 1].set_ylim(0, 390)
        axs[1, 2].set_ylim(0, 390)
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
