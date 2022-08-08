import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })
import seaborn as sns
import click

from collections import defaultdict, OrderedDict
from pathlib import Path

from src.read_write import load_json

THESIS_FOLDER = Path(__file__).resolve().parent.parent.parent
DATA_FOLDER = THESIS_FOLDER / 'data/multi_task_learning/processed'
FIGS_DIR = THESIS_FOLDER / 'results/figs'
TOYMODEL_FOLDER = THESIS_FOLDER / 'data/multi_task_learning/toymodel'

@click.command()
@click.option('--experiment', type=str,
              help='Chose experiment name (e.g. 2UHF_ICM1_ELCB1')
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--fidelities', default='uhf_hf', type=str,
              help="Chose between 'uhf_hf' or 'uhf_lf'.")
@click.option('--plot_locations', default=False, is_flag=True,
              help='Plot the locations of the samples.')
@click.option('--plot_sampling_strategy', default=False, is_flag=True,
              help='Plot the sampling strategies.')
def main(experiment, show_plots, fidelities, plot_locations,
         plot_sampling_strategy):
    plot_multitask_sample_locations_with_bincounts(
        experiment, DATA_FOLDER, plot_locations, plot_sampling_strategy,
        show_plots)


def plot_multitask_sample_locations_with_bincounts(
        experiment, folder, plot_locations, plot_sampling_strategy,
        show_plots):
    """Plot single-task sample locations in search space.

    Parameters
    ----------
    experiment : str
        e.g. 'uhf_lf_2d_elcb_strategy1_run0'
    folder : str
        e.g. 'out', path of the output files.
    """
    data = load_json(folder / f'{experiment}', '/exp_2.json')
    #parser = OutputFileParser(f'{experiment}', folder)
    samples, sample_indices = np.array(data['xy']), np.array(data['sample_indices'])
    dimension = len(samples[0]) - 1
    gmp = data['gmp']
    predicted_minimum_x = np.array(gmp)[:, 0:dimension]
    colors = ['blue' if idx_ == 0 else 'red' for idx_ in sample_indices]
    total_num_samples = len(sample_indices)
    if plot_locations:
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 3))
        alphas = np.linspace(0.1, 1.0, len(colors))[::-1]
        axs[0].scatter(samples[:, 0], samples[:, 1], c=colors, alpha=alphas)
        axs[0].set_title('Sample locations (higher intensity: earlier sample locations).\nBlue: High fidelity, Red: low fidelity',
                        fontsize=16)
        color_appearances = np.bincount(sample_indices)

        # Plot bincounts after 20, 40, 60, 80, 100% of number of samples
        bin_count_ranges = [int(quantile*total_num_samples)
                            for quantile in [0.2, 0.4, 0.6, 0.8, 1.0]]
        bin_counts = np.array([
            np.bincount(sample_indices[:num_samples])
            for num_samples in bin_count_ranges])
        axs[1].bar(range(len(bin_counts)), bin_counts[:, 0],
                    color='blue', label='HF', bottom=bin_counts[:, 1])
        axs[1].bar(range(len(bin_counts)), bin_counts[:, 1], color='red',
                    label='LF')
        plt.show()

    if plot_sampling_strategy:
        fig = plt.figure(figsize=(6.5, 3))
        colors = ['#1f78b4', '#33a02c', '#fb9a99',
                  '#e31a1c']
        for dim_idx in range(dimension):
            plt.plot(np.arange(len(predicted_minimum_x))+3,
                predicted_minimum_x[:, dim_idx],
                label=fr'$\hat{{x}}_{dim_idx}$',
                color=colors[dim_idx], lw=1.5)
            plt.scatter(range(total_num_samples), samples[:, dim_idx],
                        color=colors[dim_idx], s=30,
                        label=fr'$x_{dim_idx}$')
        for iteration_idx, sample_idx in enumerate(sample_indices):
            alpha = .2 if sample_idx == 1 else .4
            plt.axvspan(iteration_idx-.5, iteration_idx+.5,
                        facecolor='gray', alpha=alpha)
        plt.xlim(0, total_num_samples-.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_by_label = dict(zip(labels, handles))
        plt.legend(legend_by_label.values(), legend_by_label.keys(),
                   loc='center right', fontsize=10, ncol=2, framealpha=.95)
        plt.xlabel('BO Iter.')
        plt.ylabel(r'$x_i$')
        fig.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.savefig(FIGS_DIR / f'{experiment.lower()}_sampling_strategy.pdf')

if __name__ == '__main__':
    main()
