import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add path to py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
PROCESSED_DIR = THESIS_DIR / 'data/processed'
SCRIPTS_DIR = THESIS_DIR / 'scripts'

tolerances = np.array(load_yaml(SCRIPTS_DIR, '/config.yaml')['tolerances'])
TOLERANCE = tolerances[5]  # kcal/mol
TOLERANCE_IDX = np.argwhere(tolerances == TOLERANCE).squeeze()
AXIS_FONTSIZE = 15
TITLE_FONTSIZE = 15
SMALL_SIZE = 15

BLUE, RED = '#000082', '#FE0000'

PLOT_STYLE = {
            '4UHFICM1_r': {'label': 'LF➞UHF (200 initpts)',
                           'color': BLUE,
                           'linestyle': 'solid'},
            '4UHFICM3_r': {'label': 'LF➞UHF (100 initpts)',
                           'color': BLUE,
                           'linestyle': 'dashdot'},
            '4UHFICM2_r': {'label': 'HF➞UHF (200 initpts)',
                           'color': RED,
                           'linestyle': 'solid'},
            '4UHFICM4_r': {'label': 'HF➞UHF (100 initpts)',
                           'color': RED,
                           'linestyle': 'dashdot'},
            '4UHFbasic1_r': {'label': '4D Baseline',
                             'color': 'k',
                             'linestyle': 'dotted'},
            '2UHFICM1': {'label': 'LF➞UHF (50 initpts)',
                         'color': BLUE,
                         'linestyle': 'solid'},
            '2UHFICM2': {'label': 'HF➞UHF (50 initpts)',
                         'color': RED,
                         'linestyle': 'dashdot'},
            '2UHFbasic1_r': {'label': '2D Baseline',
                             'color': 'k',
                             'linestyle': 'dotted'},
            }
plt.rc('axes', labelsize=AXIS_FONTSIZE)
plt.rc('axes', titlesize=TITLE_FONTSIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


def main():
    experiment_paths = [PROCESSED_DIR.joinpath(exp) for exp in sys.argv[1:]]
    fig, axs = plt.subplots(figsize=(8, 5))
    for exp_idx, experiment_path in enumerate(experiment_paths):
        experiment_runs = [exp for exp in
                           experiment_path.iterdir() if exp.is_file()]
        data = [load_json(run, '') for run in experiment_runs]
        for (exp_run, exp_data) in zip(experiment_runs, data):
            name = str(exp_run).split('/')[-1].split('.json')[0]
            exp_data['exp_run'] = name
        data.sort(key=sort_data_by_convergence)
        plot_gmp_statistics(data, fig, axs, exp_idx)
#    plt.savefig(FIGS_DIR.joinpath('4DUHF_no_baseline.pdf'), dpi=300)
    plt.show()


def plot_gmp_statistics(data, fig, ax, idx):
    gmps = np.zeros((len(data), len(data[0]['gmp'])))
    for exp_run_idx, exp_run in enumerate(data):
        padded_array = padd_1d_array_with_zeros(
            np.array(data[exp_run_idx]['gmp'])[:, -2],
            gmps[exp_run_idx, :].shape
        )
        gmps[exp_run_idx, :] = padded_array
    gmp_mean = np.mean(gmps, axis=0)
    gmp_var = np.var(gmps, axis=0)
    # name = TITLE_DICT[data[0]["name"]]
    x_range = np.arange(gmp_mean.shape[0])
    plt.plot(x_range, gmp_mean, **PLOT_STYLE[data[0]["name"]])
    color = PLOT_STYLE[data[0]["name"]]['color']
    plt.fill_between(x_range, gmp_mean - 2*gmp_var, gmp_mean + 2*gmp_var,
                     alpha=.1, color=color)
    plt.axhline(TOLERANCE, alpha=.1, color='k', linestyle='dashed')
    plt.axhline(-TOLERANCE, alpha=.1, color='k', linestyle='dashed')
    upper_bound = 1.6 if TOLERANCE == 0.1 else 2.6
    plt.ylim(-1, upper_bound)
    if data[0]["name"] in ['2LF', '2HF', '2UHF']:
        plt.xlim(0, 30)
    plt.ylabel('GMP')
    plt.xlabel(r'Iteration $n$')
    plt.legend(fontsize=15)
    N = len(data)
    title = f'4D UHF Global Minimum Prediction (GMP) over iteration for {N} runs'
    plt.title(title)


def padd_1d_array_with_zeros(array, shape):
    """Padd an array with zeros.

    Args:
        array (np.array): original array
        shape (tuple): shape of the padded array

    Returns:
        np.array: Padded array
    """
    padded_array = np.zeros(shape)
    padded_array[:array.shape[0]] = array
    return padded_array


def sort_data_by_convergence(data):
    if data['iterations_to_gmp_convergence'][TOLERANCE_IDX] is None:
        return np.infty
    else:
        return data['iterations_to_gmp_convergence'][TOLERANCE_IDX]


if __name__ == '__main__':
    main()
