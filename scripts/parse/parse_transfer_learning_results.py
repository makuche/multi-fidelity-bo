"""
Creates pandas dataframe from TL experiments declared in 'config_tl.yaml'.
Saves the data frame as csv file.

Experiment name | initpts | dimension | iterations (0.1, 0.5, 1.0, 5.0 kcal/mol) | totaltime (0.1, 0.5, 1.0, 5.0 kcal/mol)
e.g. for 2 repeated experiments:
2HFICM1 | (2, 10) | 2 | [[8, 7, 3, 2], [13, 9, 3, 2]] | [[167.4, 123.0, 110.1, 57.9], [183.4, 97.0, 67.3, 57.9]]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from pathlib import Path
from src.read_write import load_yaml, load_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = THESIS_DIR / 'data/parsed'
# Config has (experiment,baseline) pairs
CONFIG = load_yaml(THESIS_DIR / 'scripts', '/config_tl.yaml')
TOLERANCES = CONFIG['tolerances']
names = {
    '2HFbasic1': 'HF',
    '2UHFbasic1_r': 'UHF',
    '2HFICM1': 'LF ➔ HF',
    '2UHFICM1': 'LF ➔ UHF',
    '2UHFICM2': 'HF ➔ UHF',
    '4HFbasic1': 'HF',
    '4UHFbasic1_r': 'UHF',
    '4HFICM1': 'LF ➔ HF',
    '4UHFICM1_r': 'LF ➔ UHF',
    '4UHFICM2_r': 'HF ➔ UHF',
    '4UHFICM3_r': 'LF ➔ UHF',
    '4UHFICM4_r': 'HF ➔ UHF'
}


def main():
    df = pd.DataFrame()
    dimensions = ['2D', '4D']
    for dim in dimensions:
        config = CONFIG[f'TL_experiment_plots_{dim}']
        tl_experiment_paths = [THESIS_DIR / 'data' / 'processed' /
                            exp for exp in config.keys()]
        bl_experiment_paths = [THESIS_DIR / 'data' / 'processed' /
                            baseline[0] for _, baseline in config.items()]
        bl_experiment_paths = set(bl_experiment_paths)   # Unique list
        tl_exp_data = load_experiments(tl_experiment_paths)
        bl_exp_data = load_experiments(bl_experiment_paths)
        df = create_dataframe(df, bl_exp_data, tl_exp_data)
        df.to_csv(f'{DATA_DIR}/tl_data.csv', sep=',')


def load_experiments(experiment_paths):
    all_experiments = []
    for experiment_path in experiment_paths:
        exp_data = []
        for exp in experiment_path.iterdir():
            if exp.is_file():
                exp_data.append(load_json('', exp))
        all_experiments.append(exp_data)
    return all_experiments


def create_dataframe(df, baseline_experiments, tl_experiments):
    N_bl, N_tl = len(baseline_experiments), len(tl_experiments)
    for N, exps in zip([N_bl, N_tl], [baseline_experiments, tl_experiments]):
        for exp_idx in range(N):
            exp_run_data = exps[exp_idx]
            for run_idx, run in enumerate(exp_run_data):
                # I accidentally did 30 experiments for the LF->HF experiments,
                # but as it turns out, 10 runs gives same mean statistics,
                # so it's sufficient to plot only 10 runs (also plotting
                # all 30 runs looks messy)
                if run_idx >= 10:
                    break
                name = names[run['name']]
                dim = run['dim']
                tmp_dict = {'Experiment name': name,
                            'Dimension': [dim]}
                for initpts_idx, initpts in enumerate(run['initpts']):
                    key = f'Initial points source {initpts_idx}'
                    tmp_dict[key] = initpts
                initpts = run['initpts']
                # Convergence iteration and time for all accuracies
                con_it = run['iterations_to_gmp_convergence']
                con_time = run['totaltime_to_gmp_convergence']
                for tol_idx, tol in enumerate(TOLERANCES):
                    if con_time[tol_idx] is not None:
                        con_time[tol_idx] /= 3600 
                    print(f'Totaltime [{tol} kcal/mol]', con_time[tol_idx])
                    tmp_dict[f'Iterations [{tol} kcal/mol]'] = con_it[tol_idx]
                    tmp_dict[f'Totaltime [{tol} kcal/mol]'] = con_time[tol_idx]
                row = pd.DataFrame({**tmp_dict})
                df = pd.concat([df, row], axis=0)
    df.reset_index(inplace=True, drop=True)
    return df


if __name__ == '__main__':
    main()
