import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path
from collections import OrderedDict

# For now, script is only used for the gaussian fidelity experiments
EXP_NAMES = ['UHF_B1_sobol', 'UHF_B2_manual_sobol']
EXPS_DATA = {
    'UHF_B1_sobol': OrderedDict(),
    'UHF_B2_manual_sobol': OrderedDict(),
}

THESIS_FOLDER = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_FOLDER = THESIS_FOLDER / 'data' / 'processed'
# TODO : Consider multiple exp_* files


EXP_PATHS = [PROCESSED_DATA_FOLDER / exp  for exp in EXP_NAMES]
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']


def main():
    for exp, exp_path in zip(EXP_NAMES, EXP_PATHS):
        runs = sorted([run for run in exp_path.iterdir()])
        for idx, run in enumerate(runs):
            EXPS_DATA[exp][idx] = load_json(run, '')
    energy_data = merge_data_from_manual_runs(EXPS_DATA)
    print(energy_data)
    plot_energy_over_index(energy_data)

def plot_energy_over_index(data):

    fig, ax = plt.subplots(figsize=(13,8))
    energy_shift = data['UHF_B1_sobol'][0] - data['UHF_B2_manual_sobol'][0]
    for idx, exp_name in enumerate(data):
        energies = data[exp_name]
        plt.plot(np.arange(len(energies)),
            energies, c=COLORS[idx], label=exp_name)
        if exp_name == 'UHF_B2_manual_sobol':
            energies += energy_shift
            plt.plot(np.arange(len(energies)),
                energies, c=COLORS[idx], label='shifted UHF_B2')
    plt.legend(loc='lower right', fontsize=15)
    plt.xlabel(r'iteration', fontsize=15)
    plt.ylabel(r'$E$', fontsize=15)
    plt.title(r'Calculated energy over iteration for sobol run', fontsize=15)
    plt.show()

def merge_data_from_manual_runs(data):
    exp_energies = { key: [] for key in EXP_NAMES }
    for exp_name in data:
        for exp_run in data[exp_name]:
            energies = np.array(data[exp_name][exp_run]['xy'])[::,-1]
            if 'truemin' in data[exp_name][exp_run].keys() and \
                len(data[exp_name][exp_run]['truemin']) != 0:
                energies += data[exp_name][exp_run]['truemin'][0][-1]
            for value in energies:
                exp_energies[exp_name].append(value)
    return exp_energies

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
