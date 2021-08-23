import numpy as np
import json
from pathlib import Path


PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent.parent.joinpath(
    'data/processed')

EXPS = [
    '2LFbasic1',
    '2HFbasic1',
    '2UHFbasic1',
    '4UHFbasic1_r']


def main():
    amplitude_statistics = {key: [] for key in EXPS}
    statistics_array = np.zeros((len(EXPS), 4))
    for exp in EXPS:
        statistics = get_amplitude_statistics(exp)
#        amplitude_statistics[exp].append(exp)
        for stat in statistics:
            amplitude_statistics[exp].append(round(stat, 3))

    print("Experiment, max value, min value, amplitude, variance")
    for i, exp in enumerate(amplitude_statistics.keys()):
        statistics_array[i, :] = amplitude_statistics[exp]
#        print(amplitude_statistics[exp])
    print(statistics_array)
    print (" \\\\\n".join([" & ".join(map(str,line)) for line in statistics_array]))

def load_json(path, filename):
    """
    load json file
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data


def get_amplitude_statistics(exp):
    """Calculate the amplitude over all observed acquisitions for a
    given experiment.

    Args:
        exp (str): Name of experiment

    Returns:
        amplitude: Amplitude for the given data.
    """
    exp_batch = [x for x in PROCESSED_DATA_DIR.joinpath(
               exp).iterdir() if x.is_file()]
    max_value, min_value = -np.infty, np.infty
    y_values = []
    # Search for the max and the min value over all runs and exp_batch
    for exp_run in exp_batch:
        data = load_json(exp_run, '')
        max_value = max(max_value, max(np.array(data['xy'])[:, -1]))
        min_value = min(min_value, min(np.array(data['xy'])[:, -1]))
        for y_value in (np.array(data['xy']))[:, -1]:
            y_values.append(y_value)
    amplitude = 0.5*(max_value - min_value)
    variance = np.var(np.array(y_values))
    truemin = np.array(data['truemin'])[-1, -1]
    max_value += truemin
    min_value += truemin
    return max_value, min_value, amplitude, variance

main()
