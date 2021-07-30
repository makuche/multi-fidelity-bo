import numpy as np
import json
from pathlib import Path


PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent.parent.joinpath(
    'data/processed')

EXPS = [
    '2LFbasic1',
    '2HFbasic1',
    '2UHF0basic1',
    '2UHFbasic1',
    '4UHF0basic1',
    '4UHF0basic1_r']


def main():
    amplitudes = dict()
    for exp in EXPS:
        amplitudes[exp] = get_amplitude(exp)
    for exp in amplitudes.keys():
        print(exp, amplitudes[exp])


def load_json(path, filename):
    """
    load json file
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data


def get_amplitude(exp):
    """Calculate the amplitude over all observed acquisitions for a
    given experiment.

    Args:
        exp (str): Name of experiment

    Returns:
        amplitude: Amplitude for the given data.
    """
    exp_batch = [x for x in PROCESSED_DATA_DIR.joinpath(
               exp).iterdir() if x.is_file()]
    max_amp, min_amp = -np.infty, np.infty
    # Search for the max and the min value over all runs and exp_batch
    for exp_run in exp_batch:
        data = load_json(exp_run, '')
        max_amp = max(max_amp, max(np.array(data['xy'])[:, -1]))
        min_amp = min(min_amp, min(np.array(data['xy'])[:, -1]))
    return 0.5*(max_amp - min_amp)

main()
