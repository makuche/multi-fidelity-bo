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
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config_tl.yaml')


ax_labels = [['a)', 'b)'], ['c)', 'd)']]
sub_dataframes_2D = [('2HFbasic1', '2HFICM1'),
                     ('2UHFbasic1_r', '2UHFICM1', '2UHFICM2')]
sub_dataframes_4D = [('4HFbasic1', '4HFICM1'),
                     ('4UHFbasic1_r', '4UHFICM1_r', '4UHFICM2_r', '4UHFICM3_r', '4UHFICM4_r')]
convert_strings = {
    '2HFbasic1': 'HF',
    '2UHFbasic1_r': 'UHF',
    '2HFICM1': r'LF $\rightarrow$ HF',
    '2UHFICM1': r'LF $\rightarrow$ UHF',
    '2UHFICM2': r'HF $\rightarrow$ UHF',
    '4HFbasic1': 'HF',
    '4UHFbasic1_r': 'UHF',
    '4HFICM1': r'LF $\rightarrow$ HF',
    '4UHFICM1_r': r'LF $\rightarrow$ UHF',
    '4UHFICM3_r': r'LF $\rightarrow$ UHF',
    '4UHFICM2_r': r'HF $\rightarrow$ UHF',
    '4UHFICM4_r': r'HF $\rightarrow$ UHF'
}
titles = [r'LF $\rightarrow$ HF', r'LF $\rightarrow$ UHF',
          r'HF $\rightarrow$ UHF']


@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--dimension', default='2D', type=str,
              help='Chose between 2D or 4D.')
@click.option('--tolerance', default=0.23, type=float,
              help='Tolerance level to plot convergence for.')
@click.option('--print_summary', default=False, is_flag=True,
              help='Print summary statistics of the dataframe to the terminal')
def main(show_plots, dimension, tolerance, print_summary):
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
    plot_convergence_as_boxplot(
        df, tolerance, dimension, show_plots, print_summary)


def plot_convergence_as_boxplot(df, tolerance, dimension, show_plots,
                                print_summary):

    plot_df = create_plot_dataframe(df, tolerance)
    dataframes = sub_dataframes_2D if dimension == '2D' else sub_dataframes_4D
    fig, axs = plt.subplot_mosaic([['a)', 'c)', 'c)'], ['b)', 'd)', 'd)']],
                                   figsize=(6.5, 4.5), constrained_layout=True)
    for label, ax in axs.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(
            10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=12, verticalalignment='top',
                )
    for ax_label, exp_names in zip(ax_labels, dataframes):
        names = [name for name in exp_names]
        if len(names) == 5:
            experiment_df = plot_df.loc[
                (plot_df['name'] == names[0]) | (plot_df['name'] == names[1]) |
                (plot_df['name'] == names[2]) | (plot_df['name'] == names[3]) |
                (plot_df['name'] == names[4])]
        elif len(names) == 3:
            experiment_df = plot_df.loc[
                (plot_df['name'] == names[0]) | (plot_df['name'] == names[1]) |
                (plot_df['name'] == names[2])]
        else:
            experiment_df = plot_df.loc[
                (plot_df['name'] == names[0]) | (plot_df['name'] == names[1])]
        sns.boxplot(x='Setup', y='Highest fidelity iterations',
                    data=experiment_df, ax=axs[ax_label[0]], width=0.75,
                    palette="tab10", hue='Lower fid. samples',
                    whis=[0.25, 0.75])
        sns.boxplot(x='Setup', y='CPU time [h]', data=experiment_df,
                    ax=axs[ax_label[1]], width=0.75, palette="tab10",
                    hue='Lower fid. samples', whis=[0.25, 0.75])
        if print_summary:
            print(experiment_df.groupby(['Setup', 'Lower fid. samples'])\
                ['CPU time [h]'].describe(percentiles=[.5]).round(2))
    for ax in ['a)', 'c)']:
        axs[ax].set_xlabel('')
        axs[ax].set_xticks([])
        axs[ax].legend(loc='upper right', title='Lower fid.\nsamples',
                       fontsize=10)
        handles, labels = axs[ax].get_legend_handles_labels()
        axs[ax].legend(handles=handles[:], labels=labels[:])
    for ax in ['b)', 'd)']:
        axs[ax].get_legend().remove()
        axs[ax].set_xlabel('')
    axs['b)'].set_ylabel('CPU time [h]', fontsize=14)
    axs['a)'].set_ylabel('BO iter.', fontsize=14)
    axs['c)'].set_ylabel('')
    axs['d)'].set_ylabel('')
    if show_plots:
        plt.show()
    else:
        plt.savefig(f'results/figs/transfer_learning_boxplots_{dimension}.pdf')


def create_plot_dataframe(df, tolerance):
    tolerance_idx = np.argwhere(
        tolerance == np.array(np.array(CONFIG['tolerances']))).squeeze()
    plot_df = df[['name', 'iterations_to_gmp_convergence',
                  'totaltime_to_gmp_convergence', 'initpts',
                  'cumulative_num_highest_fidelity_samples']]
    plot_df['iterations'] = plot_df[
        'iterations_to_gmp_convergence'].map(
            lambda x: chose_value_with_tolerance(x, tolerance_idx))
    plot_df['totaltime'] = plot_df[
        'totaltime_to_gmp_convergence'].map(
            lambda x: chose_value_with_tolerance(x, tolerance_idx,
                                                 time_in_h=True))
    plot_df['tl_initpts'] = plot_df[
        'initpts'].map(lambda x: x[1])
    plot_df.drop_duplicates(
        subset=['totaltime', 'tl_initpts', 'name'], inplace=True)
    plot_df['setup'] = plot_df['name'].map(lambda x: convert_strings[x])
    plot_df.rename(
        columns={'tl_initpts': 'Lower fid. samples',
                 'totaltime': 'CPU time [h]',
                 'setup': 'Setup',
                 'iterations': 'Highest fidelity iterations'},
        inplace=True)
    return plot_df


def chose_value_with_tolerance(x, tolerance_idx, time_in_h=False):
    if x[tolerance_idx] is None:
        return None
    else:
        x_tol = x[tolerance_idx] / 3600 if time_in_h else x[tolerance_idx]
        return x_tol


if __name__ == '__main__':
    main()
