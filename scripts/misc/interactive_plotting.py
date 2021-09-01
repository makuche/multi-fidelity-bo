import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add path to py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent/'data/processed'

TOLERANCE = 0.1  # kcal/mol
TOLERANCE_IDX = 5
DIM_2D = (0, 1)
DIM_4D = (0, 3)
AXIS_FONTSIZE = 15
TITLE_FONTSIZE = 15
SMALL_SIZE = 15
COLORS = ['blue', 'red', 'organe']
plt.rc('axes', labelsize=AXIS_FONTSIZE)
plt.rc('axes', titlesize=TITLE_FONTSIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 interactive_plotting.py experiments measure\n \
        Example: python3 interactive_plotting.py 2LFbasic1 gmp_stat")
        return 1
    experiment_paths = [PROCESSED_DIR.joinpath(exp) for exp in sys.argv[1:-1]]
    measure = sys.argv[-1]
    for experiment_path in experiment_paths:
        experiment_runs = [exp for exp in
                           experiment_path.iterdir() if exp.is_file()]
        data = [load_json(run, '') for run in experiment_runs]
        data.sort(key=sort_data_by_convergence)
        plot_functions = [
            plot_amplitude, plot_xy, plot_hyperparameter, plot_best_acq,
            plot_gmp_prediction, plot_xhat, plot_gmp_statistics,
            plot_gmp_multi_statistics]

        for plot_function in plot_functions:
            if measure in plot_function.__name__:
                fig, ax = plt.subplots(figsize=(8, 5))
                plot_function(data, fig, ax)
    plt.show()


def sort_data_by_convergence(data):
    if data['iterations_to_gmp_convergence'][TOLERANCE_IDX] is None:
        return np.infty
    else:
        return data['iterations_to_gmp_convergence'][TOLERANCE_IDX]


def plot_amplitude(data, fig, ax):
    for exp_run_idx, exp_run in enumerate(data):
        xy = np.array(data[exp_run_idx]['xy'])
        amps = [0.5*(max(xy[:xy_idx, -1]) - min(xy[:xy_idx, -1]))
                for xy_idx in range(1, xy.shape[0])]
        convergence = exp_run['iterations_to_gmp_convergence'][TOLERANCE_IDX]
        label = f'run {exp_run_idx+1}: convergence at {convergence}'
        plt.plot(np.arange(len(amps)), amps, label=label)

    plt.xlabel(r'Iteration $n$')
    plt.ylabel(r'GMP')
    plt.legend()
    plt.title('Amplitude over iteration')


def plot_xy(data, fig, ax):
    fig.set_size_inches(w=10, h=8)
    for exp_run_idx, exp_run in enumerate(data):
        if '4' in data[0]['name']:   # 4D runs
            bound = 150
            dim = (0, 3)
        else:                        # 2D runs
            bound = 50
            dim = (0, 1)
        xy_data = np.array(data[exp_run_idx]['xy'])[:bound, :-1]
        xy = np.empty((xy_data.shape[0], xy_data.shape[1]+1))
        xy[:, 0:xy_data.shape[1]] = xy_data
        xy[:, -1] = np.arange(1, xy_data.shape[0] + 1)
        print(xy[:5,:])
        plt.scatter(xy[:, dim[0]], xy[:, dim[1]], alpha=.7, s=20,
                    c=xy[:, -1], cmap='summer')
    plt.colorbar(label='iteration')
    N = len(data)
    title = f"""{data[0]["name"]}: Sampling locations
        for first {bound} points for {N} runs"""
    plt.title(title)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')


def plot_hyperparameter(data, fig, ax):
    pass


def plot_best_acq(data, fig, ax):
    for exp_run_idx, exp_run in enumerate(data):
        best_acq = np.array(data[exp_run_idx]['best_acq'])
        convergence = exp_run['iterations_to_gmp_convergence'][TOLERANCE_IDX]
        plt.plot(np.arange(best_acq.shape[0]), best_acq[:, -1])
        label = f'run {exp_run_idx+1}: convergence at {convergence}'
        plt.scatter(np.arange(best_acq.shape[0]), best_acq[:, -1], s=10,
                    label=label)
    plt.axhline(TOLERANCE, alpha=.3, color='k', linestyle='dashed',
                label=f'{TOLERANCE} kcal/mol tolerance')
    plt.axhline(-TOLERANCE, alpha=.3, color='k', linestyle='dashed')
    plt.ylabel('Best acquisition')
    plt.xlabel(r'Iteration $n$')
    if len(data) > 10:
        plt.legend(fontsize=10)
    else:
        plt.legend()
    plt.title(f'Best acquisition over iteration for {len(data)} run/s')


def plot_gmp_prediction(data, fig, ax):
    for exp_run_idx, exp_run in enumerate(data):
        gmp = np.array(data[exp_run_idx]['gmp'])
        convergence = exp_run['iterations_to_gmp_convergence'][TOLERANCE_IDX]
        plt.plot(np.arange(gmp.shape[0]), gmp[:, -2])
        label = f'run {exp_run_idx+1}: convergence at {convergence}'
        plt.scatter(np.arange(gmp.shape[0]), gmp[:, -2], s=10, label=label)

    plt.axhline(TOLERANCE, alpha=.3, color='k', linestyle='dashed',
                label=f'{TOLERANCE} kcal/mol tolerance')
    plt.axhline(-TOLERANCE, alpha=.3, color='k', linestyle='dashed')
    #plt.ylim(-3, 5)
    plt.ylabel('GMP')
    plt.xlabel(r'Iteration $n$')
    if len(data) > 10:
        plt.legend(fontsize=10)
    else:
        plt.legend()
    N = len(data)
    title = f'{data[0]["name"]}: GMP over iteration for {N} run/s'
    plt.title(title)


def plot_gmp_statistics(data, fig, ax):
    gmps = np.zeros((len(data), len(data[0]['gmp'])))
    for exp_run_idx, exp_run in enumerate(data):
        padded_array = padd_1d_array_with_zeros(
            np.array(data[exp_run_idx]['gmp'])[:, -2],
            gmps[exp_run_idx, :].shape
        )
        gmps[exp_run_idx, :] = padded_array

    gmp_mean = np.mean(gmps, axis=0)
    gmp_var = np.var(gmps, axis=0)
    gmp_median = np.median(gmps, axis=0)
    x_range = np.arange(gmp_mean.shape[0])
    plt.plot(x_range, gmp_mean, label=r'mean($y$)', c='blue')
    plt.plot(x_range, gmp_median, label=r'median($y$)', c='blue',
             linestyle='dashed')
    plt.fill_between(x_range, gmp_mean - 2*gmp_var, gmp_mean + 2*gmp_var,
                     alpha=.1, color='blue')
    plt.axhline(TOLERANCE, alpha=.3, color='k', linestyle='dashed',
                label='0.1 kcal/mol tolerance')
    plt.axhline(-TOLERANCE, alpha=.3, color='k', linestyle='dashed')
    plt.ylim(-1, 3)
    if data[0]["name"] in ['2LF', '2HF', '2UHF']:
        plt.xlim(0, 30)
    plt.ylabel('GMP')
    plt.xlabel(r'Iteration $n$')
    plt.legend(fontsize=15)
    N = len(data)
    title = f'{data[0]["name"]}: GMP stats over iteration for {N} run/s'
    plt.title(title)


def plot_gmp_multi_statistics(multi_data, fig, ax):
    for data_idx, data in enumerate(multi_data):
        gmps = np.zeros((len(data), len(data[0]['gmp'])))
        for exp_run_idx, exp_run in enumerate(data):
            padded_array = padd_1d_array_with_zeros(
                np.array(data[exp_run_idx]['gmp'])[:, -2],
                gmps[exp_run_idx, :].shape
            )
            gmps[exp_run_idx, :] = padded_array

        gmp_mean = np.mean(gmps, axis=0)
        gmp_var = np.var(gmps, axis=0)
        gmp_median = np.median(gmps, axis=0)
        x_range = np.arange(gmp_mean.shape[0])
        name = data[0]["name"]
        plt.plot(x_range, gmp_mean, label=f'{name} mean($y$)', c='blue')
        plt.plot(x_range, gmp_median, label=r'median($y$)', c='blue',
                linestyle='dashed')
        plt.fill_between(x_range, gmp_mean - 2*gmp_var, gmp_mean + 2*gmp_var,
                        alpha=.1, color='blue')
    plt.axhline(TOLERANCE, alpha=.3, color='k', linestyle='dashed',
                    label='0.1 kcal/mol tolerance')
    plt.axhline(-TOLERANCE, alpha=.3, color='k', linestyle='dashed')
    plt.ylim(-1, 3)
    plt.ylabel('GMP')
    plt.xlabel(r'Iteration $n$')
    plt.legend(fontsize=15)
    N = len(data)
    title = f'{data[0]["name"]}: GMP stats over iteration for {N} run/s'
    plt.title(title)


def plot_xhat(data, fig, ax):
    fig.set_size_inches(w=10, h=8)
    for exp_run_idx, exp_run in enumerate(data):
        if '4' in data[0]['name']:
            bound = 300
            dim = DIM_4D
        else:
            bound = 100
            dim = DIM_2D
        xhat_data = np.array(data[exp_run_idx]['gmp'])[:bound, :-1]
        xy = np.empty((xhat_data.shape[0], xhat_data.shape[1]+1))
        xy[:, 0:xhat_data.shape[1]] = xhat_data
        xy[:, -1] = np.arange(1, xhat_data.shape[0] + 1)
        plt.scatter(xy[:, dim[0]], xy[:, dim[1]], alpha=.9, s=10, c=xy[:, -1],
                    cmap='inferno')
    gmp_xhat = np.array(data[0]['truemin'])[0][:-1]
    plt.colorbar(label='iteration')
    plt.scatter(gmp_xhat[dim[0]], gmp_xhat[dim[1]], s=150,
                marker='*', c='green', label=r'$\hat{x}$')
    title = f"""{data[0]["name"]}: Sampling locations
        for first {bound} points for {len(data)} runs"""
    plt.title(title)
    plt.legend()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')


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


if __name__ == '__main__':
    main()
