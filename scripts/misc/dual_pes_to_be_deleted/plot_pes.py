import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import read_write

FOLDER_PATH = Path(__file__).resolve().parent
FILE_PATHS = [FOLDER_PATH / f'raw_data_{i}.json' for i in range(1, 3)]


class Settings:
    def __init__(self, data):
        self.dim = data['settings']['dim']
        self.pp_m_slice = data['settings']['pp_m_slice']
        self.bounds = np.array(data['settings']['bounds'])


def main():
    data_dicts = [read_write.load_json(path, '') for path in FILE_PATHS]
    for data_dict in data_dicts:
        data_dict['STS'] = Settings(data_dict)
        data_dict['model_data'] = np.array(data_dict['model_data'])
        data_dict['xhat'] = np.array(data_dict['xhat'])
        data_dict['acqs'] = np.array(data_dict['acqs'])
    model(*data_dicts)


def model(data_lf, data_hf):
    """
    Plots a (max 2D) slice of the model.
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    names = ['Low', 'High']
    for pes_idx, data in enumerate([data_lf, data_hf]):

        STS, model_data = data['STS'], data['model_data']
        xhat, acqs = data['xhat'], data['acqs']
        legends = True
        # xnext, acqs, legends = None, None, False
        coords = model_data[:, :STS.dim]
        mu, nu = model_data[:, -2], model_data[:, -1]
        npts = STS.pp_m_slice[2]
        x1, x2 = coords[:, STS.pp_m_slice[0]], coords[:, STS.pp_m_slice[1]]

        axs[pes_idx].contour(x1[:npts], x2[::npts],
                             mu.reshape(npts,npts), 25, colors='k')
        axs[pes_idx].set_title(f'{names[pes_idx]} fidelity')
        im = axs[pes_idx].contourf(x1[:npts], x2[::npts], mu.reshape(npts,npts),
                        150, cmap='inferno')
    cbar = fig.colorbar(im, ax=axs)
    cbar.set_label(label='$\mu(x)$', size=24)
    cbar.ax.tick_params(labelsize=18)
    lo = False
    if xhat is not None:
        plt.plot(xhat[STS.pp_m_slice[0]], xhat[STS.pp_m_slice[1]], 'r*',
        markersize=26, zorder=21, label='$\hat{x}$')
        lo = True
    if legends and lo:
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=4, mode="expand", borderaxespad=0.,
                            prop={'size':20})
        top = 0.85
    if legends:
        plt.savefig('pes.pdf', bbox_extra_artists=(lgd,),
        bbox_inches='tight')
    else:
        plt.savefig('pes.pdf')


    plt.close()

    exit()



    if acqs is not None:
        x1 = acqs[:,STS.pp_m_slice[0]]; x2 = acqs[:,STS.pp_m_slice[1]];
        sz = np.linspace(200,500,len(x1))
        lw = np.linspace(3,8,len(x1))
        plt.scatter(x1[0], x2[0], s=sz[int(len(x1)/2.)],
            linewidth=lw[int(len(x1)/2.)], zorder=10, facecolors='none',
            edgecolors='green', label='acqs')
        for i in range(len(x1)):
            plt.scatter(x1[i], x2[i], s=sz[i], linewidth=lw[i],
                zorder=10, facecolors='none', edgecolors='green')
        lo = True

    if xnext is not None:
        plt.plot(xnext[STS.pp_m_slice[0]], xnext[STS.pp_m_slice[1]], 'b^',
                        markersize=26, label='$x_{next}$', zorder=20)
        lo = True

    if minima is not None:
        x1 = minima[:,STS.pp_m_slice[0]]; x2 = minima[:,STS.pp_m_slice[1]];
        plt.scatter(x1, x2, s=350, linewidth=6, facecolors='none',
                    edgecolors='navajowhite', zorder=11, label='minima')
        lo = True


    plt.xlim(min(coords[:,STS.pp_m_slice[0]]), max(coords[:,STS.pp_m_slice[0]]))
    plt.ylim(min(coords[:,STS.pp_m_slice[1]]), max(coords[:,STS.pp_m_slice[1]]))
    plt.xlabel(axis_labels[0], size=24)
    plt.ylabel(axis_labels[1], size=24)
    top = 0.99

    plt.gcf().set_size_inches(10, 8)
    plt.gca().tick_params(labelsize=18)
    plt.gca().ticklabel_format(useOffset=False)
    #plt.tight_layout()




main()
