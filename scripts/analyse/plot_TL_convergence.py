import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import read_write


THESIS_DIR = Path(__name__).resolve().parent.parent.parent
CONFIG = read_write.load_yaml(THESIS_DIR / 'scripts' / 'analyse' / 'config'
    , '/transfer_learning.yaml')
exp_list = [THESIS_DIR / 'data' / 'processed' /
            exp for exp in CONFIG.keys()]
print(exp_list)


def load_experiments(experiment_path):
    experiment_data = []
    for exp in experiment_path.iterdir():
        if exp.is_file():
            print(exp)
            experiment_data.append(read_write.load_json('', exp))
    return experiment_data

exps = load_experiments(exp_list[0])

def plot_TL_convergence(TL_experiments, baseline_experiments, figname):

    N = len(exp_folders)
    fig, axs = plt.subplots(2, N, sigsize=(5*N, 10), sharey='row')
