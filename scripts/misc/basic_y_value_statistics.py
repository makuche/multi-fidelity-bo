import numpy as np
import json
from pathlib import Path

def load_json(path, filename):
    """
    load json file
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data
    raise FileNotFoundError(f'{path}{filename} could not be loaded with json.load')


# dict, containing <experiment>:<number of experiments> (for calculating the mean
# to get a better prediction of the amplitude over multiple experiments)
exps = {
    '2LFbasic1': 30,
    '2HFbasic1': 30,
    '2UHF0basic1': 3,
    '2UHFbasic1': 1,
    '4UHF0basic1': 1,
    '4UHF0basic1_r': 1,
    '4UHFbasic1_r': 2
}

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'

for exp in exps.keys():
    amps = []
    for i in range(exps[exp]):
        data = load_json(DATA_DIR /
                         f'processed/{exp}', f'/exp_{i+1}.json')
        max_ = max(np.array(data['xy'])[:, -1])
        min_ = min(np.array(data['xy'])[:, -1])
        amp = 0.5*(max_ - min_)
        amps.append(amp)
    mean = round(np.mean(np.array(amps)), 3)
    beta = round(2 / mean**2, 3)
    print("amps:", amps, len(amps))
    print(exp, f"mean of amplitudes over {exps[exp]} subexperiment/s: ",
         mean, "beta: ", beta)
