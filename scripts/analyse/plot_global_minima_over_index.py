import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import sys

from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)

EXP_NAMES = ['UHF', 'UHF_B2_5iterations', 'a1a1', 'a1b1']
EXP_NAMES = EXP_NAMES[:2]
OLD_RUNS = EXP_NAMES[2:]

THESIS_FOLDER = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_FOLDER = THESIS_FOLDER.joinpath('data').joinpath('processed')
EXP_PATHS = [PROCESSED_DATA_FOLDER.joinpath(exp).joinpath('exp_1.json') \
    for exp in EXP_NAMES]

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

def main():
    exps_data = dict()
    for exp, exp_path in zip(EXP_NAMES, EXP_PATHS):
        exps_data[exp] = load_json(exp_path,f'')
    #print(exps_data['UHF']['xy'])
    plot_energy_over_index(exps_data)

def plot_energy_over_index(exps_data):
    # TODO : Add Delta_E for UHF runs, so differences can become clear
    # (maybe in a extra figure)
    fig, ax = plt.subplots(figsize=(13, 8))
    for i, exp in enumerate(exps_data):
        energies = np.array(exps_data[exp]['xy'])[:,-1]
        if exp in OLD_RUNS:
            energies -= 25  # add shift for the plot
#        energies += exps_data[exp]['truemin'][0][-1]
        plt.plot(np.arange(len(energies)), energies, label=exp, c=COLORS[i])


    plt.legend(loc='lower right', fontsize=15)
    plt.xlabel(r'iteration', fontsize=15)
    plt.ylabel(r'$E$', fontsize=15)
    plt.title(r'Calculated energy over iteration for sobol run', fontsize=15)
    plt.show()

def load_json(path, filename):
    """
    load json file
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data
    raise FileNotFoundError(f'{path}{filename} could not be loaded with \
        json.load')

if __name__ == '__main__':
    main()
