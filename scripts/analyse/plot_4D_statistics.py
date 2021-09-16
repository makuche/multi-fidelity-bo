import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from read_write import load_yaml, load_json, save_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = THESIS_DIR / 'data/processed'
SCRIPTS_DIR = THESIS_DIR / 'scripts'
NAMES = ['LF', 'HF', 'UHF']


def main():

    energies = []

    data_LF = load_json(PROCESSED_DIR / '4LFbasic0', '/exp_1.json')
    data_HF = load_json(PROCESSED_DIR / '4HFbasic0', '/exp_1.json')
    data_UHF = load_json(PROCESSED_DIR / '4UHFbasic1_r', '/exp_1.json')

    energies_LF = np.array(data_LF['xy'])[:, -1]
    energies_HF = np.array(data_HF['xy'])[:, -1]
    energies_UHF = np.array(data_UHF['xy'])[:, -1]

    energies = np.array([energies_LF, energies_HF, energies_UHF])
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,5))
    axs[0].scatter(energies[0, :], energies[1, :])
    axs[1].scatter(energies[0, :], energies[2, :])
    axs[2].scatter(energies[1, :], energies[2, :])
    plt.show()
    print_correlation_matrix(energies)
    print_and_plot_summary_statistics(energies)


def print_correlation_matrix(y_values):
    """Return Pearson correlation coefficients.

    Relationship between correlation coefficient matrix R and the
    covariance matrix C is:


    Args:
        y_values (ndarray): Array containing the energies for the sobol
        experiments.

    Returns:
        correlation_matrix (ndarray): Pearson correlation coefficients
    """

    print("Correlation matrix, experiments in the order:\n", *NAMES)
    print(np.corrcoef(y_values).round(decimals=3))
    print("Covariance matrix, experiments in the order:\n", *NAMES)
    print(np.cov(y_values).round(decimals=3))


def print_and_plot_summary_statistics(energies):
    summary_statistics = np.zeros((energies.shape[0], 7))
    for exp_idx in range(energies.shape[0]):
        mean = np.mean(energies[exp_idx, :])
        max_ = max(energies[exp_idx, :])
        min_ = min(energies[exp_idx, :])
        amplitude = 0.5 * (max_ - min_)
        var = np.var(energies[exp_idx, :])
        perc_25 = np.percentile(energies[exp_idx, :], 25)
        perc_75 = np.percentile(energies[exp_idx, :], 75)
        stats = np.array((mean, var, min_, max_, amplitude, perc_25, perc_75))
        summary_statistics[exp_idx] = stats
    print("Mean, variance, min, max, amplitude, perc_25, perc_75")
    print(summary_statistics.round(2))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    axs.boxplot(energies.T, vert=True, patch_artist=True, labels=NAMES)
    axs.set_ylabel("Observed values")
    plt.show()


if __name__ == '__main__':
    main()
