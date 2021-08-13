import numpy as np
import matplotlib.pyplot as plt
#import pylustrator
# plt.switch_backend('agg')
from pathlib import Path

import read_write


class Settings:
    def __init__(self, data):
        self.dim = data['settings']['dim']
        self.pp_m_slice = data['settings']['pp_m_slice']
        self.bounds = np.array(data['settings']['bounds'])


def main():
    data = read_write.load_json(Path(__file__).resolve().parent,
                                '/raw_data.json')
    data['STS'] = Settings(data)
    del data['settings']
    data['model_data'] = np.array(data['model_data'])
    data['xhat'] = np.array(data['xhat'])
    data['acqs'] = np.array(data['acqs'])
    data['xnext'] = np.array(data['xnext'])
    model(**data)


def model(STS, dest_file, model_data, xhat=None, acqs=None, xnext=None,
                minima=None, truef=None, incl_uncert=True,
                axis_labels=None, legends=True, paths=None):
    """
    Plots a (max 2D) slice of the model.
    """
    xnext = None
    legends = False
#    pylustrator.start()
    coords = model_data[:,:STS.dim]
    mu, nu = model_data[:,-2], model_data[:,-1]
    if axis_labels is None:
        axis_labels = ["$x_%i$"%(STS.pp_m_slice[0]+1), "$x_%i$"%(STS.pp_m_slice[1]+1)]

    npts = STS.pp_m_slice[2]
    x1 = coords[:,STS.pp_m_slice[0]]
    x2 = coords[:,STS.pp_m_slice[1]]

    plt.contour(x1[:npts], x2[::npts], mu.reshape(npts,npts),
                    25, colors = 'k')
    plt.contourf(x1[:npts], x2[::npts], mu.reshape(npts,npts),
                    5, cmap='inferno') # this value is usually fixed at 150
    cbar = plt.colorbar()#, orientation='horizontal')
    cbar.set_label(label='$\mu(x)$', size=24)
    cbar.ax.tick_params(labelsize=18)

    lo = False
    if xhat is not None:
        plt.plot(xhat[STS.pp_m_slice[0]], xhat[STS.pp_m_slice[1]], 'r*',
        markersize=26, zorder=21, label='$\hat{x}$')
        lo = True

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

    if legends and lo:
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                            ncol=4, mode="expand", borderaxespad=0.,
                            prop={'size':20})
        top = 0.85
    plt.gcf().set_size_inches(10, 8)
    plt.gca().tick_params(labelsize=18)
    plt.gca().ticklabel_format(useOffset=False)
    #plt.tight_layout()


    if legends: # FOR OLD MATPLOTLIB COMPATIBILITY
        # #% start: automatic generated code from pylustrator
        # plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
        # import matplotlib as mpl
        # plt.figure(1).ax_dict["<colorbar>"].yaxis.labelpad = -45.760000
        # plt.figure(1).ax_dict["<colorbar>"].get_yaxis().get_label().set_rotation(0.0)
        # plt.figure(1).axes[0].legend(frameon=False, ncol=4, fontsize=20.0, title_fontsize=10.0)
        # plt.figure(1).axes[0].get_legend()._set_loc((-0.012903, 0.097922))
        # plt.figure(1).axes[0].get_legend()._set_loc((0.095161, 1.017001))
        # plt.figure(1).axes[0].get_xaxis().get_label().set_text("$d_{4}$")
        # plt.figure(1).axes[0].get_yaxis().get_label().set_text("$d_{13}$")
        # #% end: automatic generated code from pylustrator
#            plt.show()
        plt.tight_layout()
        plt.savefig(dest_file, bbox_extra_artists=(lgd,),
        bbox_inches='tight')
    else:
        #plt.show()
        plt.tight_layout()
        plt.savefig(dest_file)
    plt.close()

main()
