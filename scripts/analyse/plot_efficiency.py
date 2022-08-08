import matplotlib.pyplot as plt
plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })
import numpy as np
import click
import pandas as pd

from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = THESIS_DIR / 'results/tables'
FIGS_DIR = THESIS_DIR / 'results/figs'

background_colors = {'1': '#ff5b0dff', '3': '#fdbf6f', '6': '#ff7f00',
                     '50': '#a6cee3', '100': '#1f78b4', '200': '#024577'}

@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and not save) plots.')
@click.option('--dimension', default='2D', type=str,
              help='Dimension of the experiment, chose between 2D and 4D.')
def main(show_plots, dimension):
    df = load_csv_to_dataframe(TABLES_DIR / f'efficiency.csv', dimension)
    plot_efficiency(df, dimension, show_plots)


def load_csv_to_dataframe(filepath, dimension):
    df = pd.read_csv(filepath, names=['Dimension', 'Fidelities', 'TL Data',
                                      'Approach', 'Mean', 'Median'])
    df = df.sort_values(by='Mean', ascending=False)
    df = df[df['Mean'] < 100]
    return df[df['Dimension'] == dimension]


def plot_efficiency(df, dimension, show_plots):
    fig = plt.figure(figsize=(6.5, 3))
    ax = plt.gca()
    x = np.arange(len(df['Mean']))
    plt.scatter(x, df['Mean'], label='Mean', c='k', marker='x', zorder=10)
    plt.plot(x, df['Mean'],  c='k', zorder=8)
    plt.scatter(x, df['Median'], label='Median', c='k', marker='o',
                zorder=9)
    labels = df['Fidelities']
    for idx, approach in enumerate(df['Approach']):
        if np.isnan(approach):
            continue
        approach = str(int(approach))
        plt.axvspan(idx-.5, idx+.5, facecolor=background_colors[approach],
                    alpha=.7, label=f'MFBO {approach}')
    for idx, approach in enumerate(df['TL Data']):
        if np.isnan(approach):
            continue
        approach = str(int(approach))
        plt.axvspan(idx-.5, idx+.5, facecolor=background_colors[approach],
                    alpha=.7, label=f'TL ({approach})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels=labels, rotation=45)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, ncol=2,
              loc='upper right')
    ax.set_xlim(x[0]-.5, x[-1]+.5)
    plt.ylabel(r'Relative cost [\%]')

    #add annotation
    fig.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.savefig(FIGS_DIR / f'efficiency_{dimension}.pdf')


if __name__ == '__main__':
    main()
