from typing import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from collections import OrderedDict
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json


THESIS_DIR = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_DIR / 'results/figs'
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/MT_config.yaml')

BLUE, RED = '#000082', '#FE0000'
SCATTER_DICT = {'color': BLUE, 'alpha': .4, 'marker': 'x',
                'label': 'observation', 's': 60}
SCATTER_DICT_BOXPLOT = {'color': BLUE, 'alpha': .2, 'marker': 'x',
                'label': 'observation', 's': 30}
MEANS_DICT = {'color': RED, 'marker': '*', 's': 120, 'label': 'mean'}


def main():
    # config = CONFIG[f'TL_experiment_plots_2D']
    config = CONFIG[f'TL_experiment_plots_4D']
    # tl_experiments = [THESIS_DIR / 'data' / 'processed' / 'MT_a3b7']
    tl_experiments = [THESIS_DIR / 'data' / 'processed' / 'MT_b3b1']
    bl_experiments = [THESIS_DIR / 'data' / 'processed' / config[exp][0]
                      for exp in config]

    raw_tl_data = load_raw_experiment_data(tl_experiments)
    raw_bl_data = load_raw_experiment_data(bl_experiments)

    tl_data_dict = load_values_to_dict(raw_tl_data)
    bl_data_dict = load_values_to_dict(raw_bl_data)

    # plot_TL_convergence('MT_a3b7_linearreg.png', tl_data_dict, bl_data_dict,
    #                      tol_idx=5, show_plots=True)
    plot_TL_convergence('MT_b3b1_linearreg.png', tl_data_dict, bl_data_dict,
                         tol_idx=5, show_plots=True)
    # plot_TL_boxplot('MT_a3b7_boxplot.png', tl_data_dict, bl_data_dict,
    #                 tol_idx=5)


def load_raw_experiment_data(experiments):
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        for exp in experiment.iterdir():
            if exp.is_file():
                exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def load_values_to_dict(exp_list, tol_idx=5):
    data_dict = OrderedDict()
    for exp_idx, exp in enumerate(exp_list):
        data = OrderedDict()
        for exp_run_idx, exp_run in enumerate(exp):
            data[str(exp_run_idx+1)] = tmp = {}
            tmp['initpts'] = exp_run['initpts']
            tmp['iterations_to_gmp_convergence'] = exp_run['iterations_to_gmp_convergence']
            tmp['totaltime_to_gmp_convergence'] = exp_run['totaltime_to_gmp_convergence']
        data_dict[str(exp_idx+1)] = data
    return data_dict


def plot_TL_convergence(figname, data_tl, data_bl, tol_idx, show_plots=False):
    N = len(data_tl)
    fig, axs = plt.subplots(2, N, figsize=(3*N + 1, 6), sharex=True)

    for tl_exp_idx in range(N):
        tl_exp_data = data_tl[str(tl_exp_idx+1)]
        bl_exp_data = data_bl[str(tl_exp_idx+1)]

        tl_bo_data, bl_bo_data = {}, {}
        for exp_run_key in tl_exp_data:
            data = tl_exp_data[exp_run_key]
            initpts  = data['initpts'][1]
            if str(initpts) not in tl_bo_data:
                tl_bo_data[str(initpts)] = {'iterpts': [], 'time': []}
            iterpts = data['iterations_to_gmp_convergence'][tol_idx]
            time = data['totaltime_to_gmp_convergence'][tol_idx]
            axs[0].scatter(initpts, iterpts, **SCATTER_DICT)
            axs[1].scatter(initpts, time, **SCATTER_DICT)
            tl_bo_data[str(initpts)]['iterpts'].append(iterpts)
            tl_bo_data[str(initpts)]['time'].append(time)
        for bl_exp_run_key in bl_exp_data:
            data = bl_exp_data[bl_exp_run_key]
            initpts = data['initpts'][1]
            if str(initpts) not in bl_bo_data:
                bl_bo_data[str(initpts)] = {'iterpts': [], 'time': []}
            iterpts = data['iterations_to_gmp_convergence'][tol_idx]
            time = data['totaltime_to_gmp_convergence'][tol_idx]
            axs[0].scatter(initpts, iterpts, **SCATTER_DICT)
            axs[1].scatter(initpts, time, **SCATTER_DICT)
            bl_bo_data[str(initpts)]['iterpts'].append(iterpts)
            bl_bo_data[str(initpts)]['time'].append(time)
        iterpts_means = {key: np.mean(tl_bo_data[key]['iterpts'])
                         for key in tl_bo_data}
        times_means = {key: np.mean(tl_bo_data[key]['time'])
                       for key in tl_bo_data}
        for key in bl_bo_data:
            iterpts_means[str(key)] = np.mean(bl_bo_data[key]['iterpts'])
            times_means[str(key)] = np.mean(bl_bo_data[key]['time'])
        # Linear Regression
        sorted_iterpts_keys = sorted([int(key) for key in iterpts_means])
        train_data_iterpts = np.array([(key, iterpts_means[str(key)])
                               for key in sorted_iterpts_keys])
        reg = LinearRegression().fit(train_data_iterpts[:, 0].reshape(-1, 1),
                                     train_data_iterpts[:, 1].reshape(-1, 1))
        y_predict = reg.predict(train_data_iterpts[:, 0].reshape(-1, 1))
        axs[0].plot(train_data_iterpts[:, 0], y_predict,
                    color='red', linewidth=3, zorder=3)

        train_data_times = np.array([(key, times_means[str(key)])
                                     for key in sorted_iterpts_keys])
        reg = LinearRegression().fit(train_data_times[:, 0].reshape(-1, 1),
                                     train_data_times[:, 1].reshape(-1, 1))
        print(reg.coef_, reg.intercept_)
        y_predict = reg.predict(train_data_times[:, 0].reshape(-1, 1))
        axs[1].plot(train_data_times[:, 0], y_predict,
                    color='red', linewidth=3, zorder=3)

        for iterpts in iterpts_means:
            axs[0].scatter(int(iterpts), iterpts_means[iterpts], **MEANS_DICT,
                           zorder=5)
            axs[1].scatter(int(iterpts), times_means[iterpts], **MEANS_DICT,
                           zorder=5)
    axs[1].set_xlabel('Secondary initpoints')
    axs[1].set_ylabel('CPU time [s]')
    axs[0].set_ylabel('BO iterations')
    fig.tight_layout()

    # This was for a3b7
    # nuutti_reg_coefs_iterpts = np.array([-0.19230303, 21.08333333])
    # nuutti_means_iterpts = np.array(
    #                     [[0.0, 24.0],
    #                     [5.0, 19.8],
    #                     [10.0, 15.0],
    #                     [15.0, 15.1],
    #                     [20.0, 14.0],
    #                     [25.0, 16.8],
    #                     [30.0, 12.3],
    #                     [35.0, 11.3],
    #                     [40.0, 17.1],
    #                     [45.0, 13.4],
    #                     [50.0, 14.4]])
    # axs[0].scatter(nuutti_means_iterpts[:, 0], nuutti_means_iterpts[:, 1], color='green',
    #                zorder=5, marker='*', s=120, label='a3b7')
    # x_pred = np.linspace(0, 50)
    # axs[0].plot(x_pred,
    #     nuutti_reg_coefs_iterpts[0]*x_pred + nuutti_reg_coefs_iterpts[1],
    #     color='green', linewidth=3, zorder=3)

    # nuutti_reg_coefs_times = np.array([-1.66288686, 585.24235278])
    # nuutti_means_times = np.array(
    #                             [[0.0, 645.4350000000001],
    #                             [5.0, 597.1326999999999],
    #                             [10.0, 477.47119999999995],
    #                             [15.0, 485.60069999999996],
    #                             [20.0, 468.88289999999995],
    #                             [25.0, 559.6504000000001],
    #                             [30.0, 464.8541],
    #                             [35.0, 448.9845],
    #                             [40.0, 611.0294999999999],
    #                             [45.0, 525.0977],
    #                             [50.0, 575.848]])
    # axs[1].scatter(nuutti_means_times[:, 0], nuutti_means_times[:, 1], color='green',
    #                zorder=5, marker='*', s=120, label='a3b7')
    # x_pred = np.linspace(0, 50)
    # axs[1].plot(x_pred,
    #     nuutti_reg_coefs_times[0]*x_pred + nuutti_reg_coefs_times[1],
    #     color='green', linewidth=3, zorder=3)

    # This is for b3b1
    nuutti_reg_coefs_iterpts = np.array([-0.37930032, 140.314377])
    nuutti_means_iterpts = np.array(
                            [[0.0, 140.06666666666666],
                            [10.0, 141.4],
                            [20.0, 143.5],
                            [30.0, 145.5],
                            [40.0, 128.1],
                            [50.0, 115.0],
                            [60.0, 118.3],
                            [70.0, 120.7],
                            [80.0, 105.4],
                            [90.0, 102.6],
                            [100.0, 85.7],
                            [110.0, 85.4],
                            [120.0, 81.6],
                            [130.0, 88.9],
                            [140.0, 65.3],
                            [150.0, 71.2],
                            [160.0, 80.9],
                            [170.0, 89.6],
                            [180.0, 84.5],
                            [190.0, 83.6],
                            [200.0, 73.3]])

    axs[0].scatter(nuutti_means_iterpts[:, 0], nuutti_means_iterpts[:, 1], color='green',
                   zorder=5, marker='*', s=120, label='a3b7')
    x_pred = np.linspace(0, 200)
    axs[0].plot(x_pred,
        nuutti_reg_coefs_iterpts[0]*x_pred + nuutti_reg_coefs_iterpts[1],
        color='green', linewidth=3, zorder=3)

    nuutti_reg_coefs_times = np.array([1.30012091, 5044.04466933])
    nuutti_means_times = np.array(
                                [[0.0, 4999.187099999999],
                                [10.0, 5888.4130000000005],
                                [20.0, 5931.570699999999],
                                [30.0, 6138.2623],
                                [40.0, 5298.0442],
                                [50.0, 4738.927900000001],
                                [60.0, 5043.1826],
                                [70.0, 5286.5959],
                                [80.0, 4639.3949999999995],
                                [90.0, 4769.5546],
                                [100.0, 4130.161600000001],
                                [110.0, 4199.3614],
                                [120.0, 4256.906000000001],
                                [130.0, 4918.1248],
                                [140.0, 4020.2249],
                                [150.0, 4508.5657],
                                [160.0, 5297.2916],
                                [170.0, 5954.224],
                                [180.0, 6230.1858],
                                [190.0, 6277.823200000001],
                                [200.0, 6218.904799999999]])
    axs[1].scatter(nuutti_means_times[:, 0], nuutti_means_times[:, 1], color='green',
                   zorder=5, marker='*', s=120, label='a3b7')
    x_pred = np.linspace(0, 200)
    axs[1].plot(x_pred,
        nuutti_reg_coefs_times[0]*x_pred + nuutti_reg_coefs_times[1],
        color='green', linewidth=3, zorder=3)
    if show_plots:
        plt.show()
    plt.savefig(figname)


def plot_TL_boxplot(figname, data_tl, data_bl, tol_idx, show_plots=False):
    N = len(data_tl)
    fig, axs = plt.subplots(2, N, figsize=(4*N + 4, 8), sharex=True)

    for tl_exp_idx in range(N):
        tl_exp_data = data_tl[str(tl_exp_idx+1)]
        bl_exp_data = data_bl[str(tl_exp_idx+1)]

        tl_bo_data, bl_bo_data = {}, {}
        for exp_run_key in tl_exp_data:
            data = tl_exp_data[exp_run_key]
            initpts  = data['initpts'][1]
            if initpts not in tl_bo_data:
                tl_bo_data[initpts] = {'iterpts': [], 'time': []}
            iterpts = data['iterations_to_gmp_convergence'][tol_idx]
            time = data['totaltime_to_gmp_convergence'][tol_idx]
            axs[0].scatter(initpts, iterpts, **SCATTER_DICT_BOXPLOT)
            axs[1].scatter(initpts, time, **SCATTER_DICT_BOXPLOT)
            tl_bo_data[initpts]['iterpts'].append(iterpts)
            tl_bo_data[initpts]['time'].append(time)
        for bl_exp_run_key in bl_exp_data:
            data = bl_exp_data[bl_exp_run_key]
            initpts = data['initpts'][1]
            if initpts not in bl_bo_data:
                bl_bo_data[initpts] = {'iterpts': [], 'time': []}
            iterpts = data['iterations_to_gmp_convergence'][tol_idx]
            time = data['totaltime_to_gmp_convergence'][tol_idx]
            axs[0].scatter(initpts, iterpts, **SCATTER_DICT_BOXPLOT)
            axs[1].scatter(initpts, time, **SCATTER_DICT_BOXPLOT)
            bl_bo_data[initpts]['iterpts'].append(iterpts)
            bl_bo_data[initpts]['time'].append(time)
        iterpts_means = {key: np.mean(tl_bo_data[key]['iterpts'])
                         for key in tl_bo_data}
        times_means = {key: np.mean(tl_bo_data[key]['time'])
                       for key in tl_bo_data}
        for key in bl_bo_data:
            iterpts_means[key] = np.mean(bl_bo_data[key]['iterpts'])
            times_means[key] = np.mean(bl_bo_data[key]['time'])
        tl_bo_data = OrderedDict(sorted(tl_bo_data.items(), key=lambda x: int(x[0])))
        xticks = [key for key in tl_bo_data]
        xticks.insert(0, 0)
        bo_boxplot_data = [tl_bo_data[key]['iterpts'] for key in tl_bo_data]
        bo_boxplot_data.insert(0, bl_bo_data[0]['iterpts'])
        cpu_boxplot_data = [tl_bo_data[key]['time'] for key in tl_bo_data]
        cpu_boxplot_data.insert(0, bl_bo_data[0]['time'])
        axs[0].boxplot(bo_boxplot_data, positions=xticks, meanline=True, whis=(0, 100),
                       widths= [2.5] * len(bo_boxplot_data), zorder=10,
                       medianprops={'lw': 2, 'c': RED})
        axs[1].boxplot(cpu_boxplot_data, positions=xticks, meanline=True, whis=(0, 100),
                       widths= [2.5] * len(cpu_boxplot_data), zorder=10,
                       medianprops={'lw': 2, 'c': RED})
        axs[0].set_xticks(xticks)
        axs[1].set_xticks(xticks)
    axs[1].set_xlabel('Secondary initpoints')
    axs[1].set_ylabel('CPU time [s]')
    axs[0].set_ylabel('BO iterations')
    fig.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(figname)

if __name__ == '__main__':
    main()
