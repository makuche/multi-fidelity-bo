import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = THESIS_DIR / 'results/figs'
RAW_DATA_PATH = THESIS_DIR / 'data/raw'
BASELINE_RUNS = ['2LFbasic1/exp_1/', '2HFbasic1/exp_1/', '2UHFbasic1/exp_1/']

FILE_PATHS = [RAW_DATA_PATH / file / 'postprocessing/raw_data_for_plot.json'
              for file in BASELINE_RUNS]
NAMES = ['Low', 'High', 'Ultra high']
TRUEMINS_2D = {'Low':         17.4815,
               'High':       -203012.37364,
               'Ultra high': -202861.33811}


class Settings:
    def __init__(self, data):
        self.dim = data['settings']['dim']
        self.pp_m_slice = data['settings']['pp_m_slice']
        self.bounds = np.array(data['settings']['bounds'])


def main():
    data_dicts = [load_json(path, '') for path in FILE_PATHS]
    for data_dict in data_dicts:
        data_dict['STS'] = Settings(data_dict)
        data_dict['model_data'] = np.array(data_dict['model_data'])
        data_dict['xhat'] = np.array(data_dict['xhat'])
        data_dict['acqs'] = np.array(data_dict['acqs'])
    model(*data_dicts)
    #model_differences(*data_dicts)


def model(data_lf, data_hf, data_uhf):
    """
    Plots a (max 2D) slice of the model.
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 7),
                            sharex=True, sharey=True)
    for pes_idx, data in enumerate([data_lf, data_hf, data_uhf]):
        STS, model_data = data['STS'], data['model_data']
        print("STS.dim", STS.dim)
        data['model_data'][:, -2] -= TRUEMINS_2D[NAMES[pes_idx]]
        xhat, acqs = data['xhat'], data['acqs']
        coords = model_data[:, :STS.dim]
        mu, nu = np.sqrt(model_data[:, -2]), model_data[:, -1]
        npts = STS.pp_m_slice[2]
        x1, x2 = coords[:, STS.pp_m_slice[0]], coords[:, STS.pp_m_slice[1]]
        print("STS.pp", STS.pp_m_slice[0], STS.pp_m_slice[1], STS.pp_m_slice[2])
        #amplitude = max(mu) - min(mu)
        #mu /= amplitude
        axs[pes_idx].contour(x1[:npts], x2[::npts],
                             mu.reshape(npts, npts), 15, colors='k')
        im = axs[pes_idx].contourf(x1[:npts], x2[::npts],
                                   mu.reshape(npts, npts), 50, cmap='viridis')
        if pes_idx == 0:
            axs[pes_idx].scatter(*xhat, c='red', marker='*', s=150, label='GMP')
        else:
            axs[pes_idx].scatter(*xhat, c='red', marker='*', s=150)
    fig.colorbar(im, ax=axs, shrink=0.8, location='bottom')
    axs[0].legend(fontsize=15)
    txt = axs[0].text(-45, 285, '(a) LF', c='white',
                      weight='bold', fontsize=15)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
    txt = axs[1].text(-45, 285, '(b) HF', c='white',
                      weight='bold', fontsize=15)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
    txt = axs[2].text(-45, 285, '(c) UHF', c='white',
                      weight='bold', fontsize=15)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98,
                        bottom=0.28, wspace=0.05)
    plt.show()
    plt.close()


def model_differences(data_lf, data_hf, data_uhf):
    for pes_idx, data in enumerate([data_lf, data_hf, data_uhf]):
        data['model_data'][:, -2] -= TRUEMINS_2D[NAMES[pes_idx]]
    energy_diff_lf_hf = data_lf['model_data'][:, -2] - data_hf['model_data'][:, -2]
    energy_diff_lf_uhf = data_lf['model_data'][:, -2] - data_uhf['model_data'][:, -2]
    energy_diff_hf_uhf = data_hf['model_data'][:, -2] - data_uhf['model_data'][:, -2]

    for energy_diff in [energy_diff_lf_hf, energy_diff_lf_uhf, energy_diff_hf_uhf]:
        tmp = copy.deepcopy(data_lf['model_data'])
        diff_lf_hf = None # TODO

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 7),
                            sharex=True, sharey=True)
    names = ['Low', 'High', 'Ultra high']
    for pes_idx, data in enumerate([diff_lf_hf, diff_lf_uhf, diff_hf_uhf]):
        npts = 100
        print(data.shape)
        coords = data[:, :2]
        mu, nu = data[:, -2], data[:, -1]
        x1, x2 = coords[:, 0], coords[:, 1]
        axs[pes_idx].contour(x1[:npts], x2[::npts],
                             mu.reshape(npts, npts), 15, colors='k')
        axs[pes_idx].contourf(x1[:npts], x2[::npts],
                              mu.reshape(npts, npts), 150, cmap='viridis')
    plt.show()
    plt.close()
main()
