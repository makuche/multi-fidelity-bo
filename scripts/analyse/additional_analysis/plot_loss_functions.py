import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from read_write import load_yaml, load_json, save_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs/'



# Loss functions minima
# Experiment | Secondary initpts | Min. loss
loss_data_minimum = {
             '2D LF➔HF': [35, 0.70],
             '2D LF➔VHF': [35, 0.72],
             '2D LF➔UHF': [50, 0.76],
             '2D HF➔VHF': [20, 0.68],
             '2D HF➔UHF': [50, 0.27],
             '4D LF➔HF': [140, 0.8],
             '4D LF➔VHF': [110, 0.6],
             '4D LF➔UHF': [200, 0.47],
             '4D HF➔VHF': [170, 0.66],
             '4D HF➔UHF': [200, 0.31]
}
COLORS = {'2D': '#000082', '4D': '#FE0000'}
MEDIUM_FONTSIZE = 15
INDICATOR_LOSS_STYLE = {'0': {'color': '#3EE1D1', 'label': 'slower than baseline'},
                        '1': {'color': '#FF8C00', 'label': 'faster than baseline'}}
LABEL_DICT = {
    'a3b7': '2D LF➔HF',
    'a3c3': '2D LF➔VHF',
    'a3c4': '2D HF➔VHF',
    'b3b1': '4D LF➔HF',
    'b3c1': '4D LF➔VHF',
    'b3c2': '4D HF➔VHF',
    '2UHFICM1': '2D LF➔UHF',
    '2UHFICM2': '2D HF➔UHF',
    '4UHFICM1': '4D LF➔UHF',
    '4UHFICM2': '4D HF➔UHF'
}

PLOT_ORDER = {'a3b7': 0, 'a3c3': 1, 'a3c4': 2, 'b3b1': 5, 'b3c1': 6,
              'b3c2': 7, '2UHFICM1': 3, '2UHFICM2': 4, '4UHFICM1': 8,
              '4UHFICM2': 9}

def main():
    loss_data = load_yaml(THESIS_DIR / 'results/tables', '/loss_table.yaml')
    plot_loss_data_minimum(loss_data_minimum)
    plt.rc('xtick', labelsize=16)
    plot_indicator_loss(loss_data)


def plot_loss_data_minimum(data):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for key in data:
        data[key] = np.array(data[key])
        plt.scatter(*data[key], c=COLORS[key[:2]], label=key[:2])
        if '2D HF➔VHF' in key:
            shift = data[key] + np.array([-22, -0.042])
        elif '2D LF➔VHF' in key:
            shift = data[key] + np.array([15, -0.015])
        elif '2D LF➔HF' in key:
            shift = data[key] + np.array([15, -0.035])
        else:
            shift = data[key] + np.array([-22, 0.012])
        ax.text(*shift, key[2:], c='black', fontsize=16)

    # Small dash for overlapping results
    plt.plot([38, 51], [0.7192, 0.722], c='black')
    plt.plot([38, 51], [0.6992, 0.692], c='black')

    # This disables duplicates in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=16)

    plt.title('Reduction of CPU time (relative to baseline)',
              fontsize=MEDIUM_FONTSIZE)
    plt.xlabel('Secondary initialization points', fontsize=MEDIUM_FONTSIZE)
    plt.ylabel('Mean loss', fontsize=MEDIUM_FONTSIZE)
    ax.set_xlim(-10, 230)
    ax.set_ylim(0, 1)
    # plt.show()
    plt.savefig(FIGS_DIR / 'mean_loss_function.png', dpi=300)


def plot_indicator_loss(data):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    data = data['TL_experiments']
    data_dict = {}
    for exp in data:
        if exp[0] not in data_dict:
            data_dict[exp[0]] = []
        data_dict[exp[0]].append(exp[1:])

    for exp_name in data_dict:
        for exp_run in data_dict[exp_name]:
            x, indicator_loss = exp_run[0], exp_run[-1]
            plt.scatter(x, PLOT_ORDER[exp_name],
                        **INDICATOR_LOSS_STYLE[str(indicator_loss)], s=90)

    y_labels = {k: v for k, v in
                sorted(PLOT_ORDER.items(), key=lambda item: item[1])}
    y_labels = [key for key in y_labels]
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=16,
               loc='lower right')
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=12)
    plt.title('Indicator loss', fontsize=16)
    ax.set_xticks([0, 25, 50, 100, 150, 200])
    plt.xlabel('Secondary initialization points', fontsize=14)
    plt.savefig(FIGS_DIR / 'indicator_loss_function.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
