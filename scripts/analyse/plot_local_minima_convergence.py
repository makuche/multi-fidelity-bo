import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO : Maybe take this to the interactive plotting script

cwd = os.getcwd()                    # current working directory
# folder name, containing the data files
# TODO : Fix this with patlib
MINIMA_DATA_LOCATION = '/home/manuel/Dropbox/Studium/Master/Thesis Project/thesis/results/figs/pes/UHF_BOSS_run/postprocessing/data_local_minima/'
RESULTS_LOCATION = '/home/manuel/Dropbox/Studium/Master/Thesis Project/thesis/results/figs/'
#pes_data_folder = '/data_pes/'

colors = ['#00ffff', '#0ef1ff', '#1ce3ff', '#2ad4ff', '#39c6ff', '#47b8ff',
'#55aaff', '#639cff', '#718eff', '#8080ff', '#8e71ff', '#9c63ff', '#aa55ff',
'#b847ff', '#c639ff', '#d42bff', '#e31cff', '#f10eff', '#ff00ff']

linestyles = ['solid', 'dotted', 'dashed', 'dashdot']

def main():
    data_minima, data_pes = get_filenames(MINIMA_DATA_LOCATION)
    data_files_local_minima = [load_data_into_array(data_minima[i], \
        MINIMA_DATA_LOCATION, '') for i in range(len(data_minima))]
    #data_files_pes = [load_data_into_array(data_pes[i], pes_data_folder, '') for i in range(len(data_pes))]
    #plot_convergence_over_iteration(data_files_local_minima, data_files_pes)
    plot_convergence_over_iteration(data_files_local_minima)

def get_filenames(minima_data_folder):
    entries = os.listdir(minima_data_folder)
    entries = sorted(entries)
    #entries_pes = os.listdir('data_pes/')
    entries_pes = None
    return entries, entries_pes

def load_data_into_array(file_name, data_folder, cwd):
    file_path = str(cwd + data_folder + file_name)
    data = np.loadtxt(file_path)
    return data


def plot_convergence_over_iteration(list_containing_minima, data_pes=None):
    energies_list, coordinates_list = [], []
    fig, ax = plt.subplots(figsize=(13, 8))
    for minima in list_containing_minima:
        minima = np.array(minima)
        if len(minima.shape) == 1:
            minima = np.expand_dims(minima, axis=0)
        coordinates = minima[:,0:2]
        energies = minima[:,2]
        energies_list.append(energies)
        coordinates_list.append(coordinates)

    plt_idx = 0
    for idx, energy in enumerate(energies_list):
        if 5*idx <= 15:
            print("Skipping", 5*idx)
            continue
        print("Plotting", 5*idx)
        plt_idx += 1
        ls = 'solid' if 5*idx >= 45 else 'dashed'
        plt.plot(np.arange(len(energy))+1, energy, label=str(5*idx),
            color=colors[plt_idx % len(colors)], linestyle=ls, linewidth=2)
                # linestyle=linestyles[plt_idx % len(linestyles)])
    plt.xlabel(r'$i$-th local minima', fontsize=15)
    ax.set_xticks([1,2,3,4,5])
    plt.ylabel(r'E', fontsize=15)
    plt.title(r'Local minima after $n$ BOSS iterations', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.savefig(RESULTS_LOCATION + 'ordered_minima_predictions.pdf')

    if data_pes is not None:
        plt.subplots(figsize=(13, 8))
        pes = np.array(data_pes[0])     #pes shape is (x,y,mu(energy),nu(uncertainty))
        x, y, E = pes[:,0], pes[:,1], pes[:,2]
        #plt.tricontourf(x, y, E, 150, cmap='viridis')
        contourplot = plt.tricontour(x, y, E, 25, cmap='viridis')
        plt_idx = 0
        for idx, coordinates in enumerate(coordinates_list):
            if len(coordinates) > 2 and len(coordinates) < 7:
                if idx % 5 == 0:
                    plt_idx += 1
                    marker = 'o' if idx > 45 else '.'
                    plt.scatter(coordinates[:,0], coordinates[:,1], label=str(idx), s=80,
                    color=colors[plt_idx % len(colors)], marker=marker)
        plt.legend()
        plt.savefig(RESULTS_LOCATION + 'minima_locations.pdf')

main()
