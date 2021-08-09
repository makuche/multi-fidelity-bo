import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import read_write


THESIS_DIR = Path(__name__).resolve().parent.parent.parent
TL_CONFIG = read_write.load_yaml(THESIS_DIR / 'scripts/analyse/config',
                              '/transfer_learning.yaml')
tl_experiments = [THESIS_DIR / 'data' / 'processed' /
                  exp for exp in TL_CONFIG.keys()]
# baselines, corresponding to each tl experiment
baseline_experiments = [THESIS_DIR / 'data' / 'processed' /
                  TL_CONFIG[exp][0] for exp in TL_CONFIG]


def load_experiments(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        for exp in experiment.iterdir():
            if exp.is_file():
                exp_data.append(read_write.load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


tl_experiment_data = load_experiments(tl_experiments)
baseline_experiment_data = load_experiments(baseline_experiments)


def plot_tl_convergence(figname, baseline_experiments, tl_experiments):

    # TODO : Remove outliers for y axis values by setting them to NaN
    N = len(tl_experiments)
    fig, axs = plt.subplots(2, N, figsize=(5*N, 10), sharey='row')
    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    LARGE_SIZE = 25

    for i in range(N):
        baseline_data = baseline_experiments[i]
        tl_data = tl_experiments[i]

        explist = baseline_data
        for tl_exp in tl_data:
            explist.append(tl_exp)

        convergence_iterations, convergence_times = [], []
        for exp in explist:
            if len(exp['initpts']) > 1:
                secondary_initpts = int(exp['initpts'][1])
            else:
                secondary_initpts = 0
            convergence_iter = exp['iterations_to_gmp_convergence'][5]
            convergence_iterations.append([secondary_initpts,
                                           convergence_iter])

            convergence_time = exp['totaltime_to_gmp_convergence'][5]
            convergence_times.append([secondary_initpts,
                                      convergence_time])

        # scatter plot
        convergence_iterations = np.array(convergence_iterations, dtype=float)
        print(convergence_iterations)
        axs[0,i].scatter(convergence_iterations[:,0],
                         convergence_iterations[:,1],
                         color='blue', alpha=.5, marker='x',
                         label='observation')

        # fit
        x = convergence_iterations[:,0].reshape(-1,1)
        y = convergence_iterations[:,1].reshape(-1,1)
        reg = LinearRegression().fit(x,y)
        x_plot = np.arange(0, 50, 0.01).reshape(-1,1)
        y_plot = reg.predict(x_plot)
        axs[0,i].plot(x_plot, y_plot, color='red', label='trend', linewidth=3)

        # means
        for initpts in np.unique(x):
            mean = np.mean(y[x == initpts])
            axs[0, i].scatter([initpts], [mean], color='red', marker='s',
                              label='mean')
        if i == N-1:
            axs[0, N-1].legend(fontsize=SMALL_SIZE)

        # scatter plot
        convergence_times = np.array(convergence_times, dtype=float)
        axs[1,i].scatter(convergence_times[:,0],
                         convergence_times[:,1],
                         color='blue', alpha=.5, marker='x',
                         label='observation')

        # fit
        x = convergence_times[:, 0].reshape(-1, 1)
        y = convergence_times[:, 1].reshape(-1, 1)
        reg = LinearRegression().fit(x,y)
        x_plot = np.arange(0, 50, 0.01).reshape(-1, 1)
        y_plot = reg.predict(x_plot)
        axs[1,i].plot(x_plot, y_plot, color='red', label='trend', linewidth=3)

        expname = tl_experiments[i][0]['name'].split('_')[0]
        title = f'{i+1}a) {expname}'
        axs[0,i].set_title(title, loc='left', fontsize=MEDIUM_SIZE)
        title = f'{i+1}b) {expname}'
        axs[1,i].set_title(title, loc='left', fontsize=MEDIUM_SIZE)

    axs[0,0].set_ylabel('BO iterations to GMP convergence',
                        fontsize=SMALL_SIZE)
    axs[1,0].set_ylabel('CPU time to GMP convergence', fontsize=SMALL_SIZE)


    for ax in axs[1,:]:
        ax.set_xlabel('secondary initpts', fontsize=SMALL_SIZE)

    for ax in axs.flatten():
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params('x', labelrotation=40)
        ax.tick_params(axis='both',
                       width=3, length=4,
                       labelsize=SMALL_SIZE)

    plt.show()


plot_tl_convergence('', baseline_experiment_data, tl_experiment_data)
