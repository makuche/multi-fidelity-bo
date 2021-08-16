import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from numpy.core.numeric import ones_like

import read_write

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / 'data/processed'

#COLORS = ['#440154', '#46085c', '#471063', '#481769', '#481d6f', '#482475', '#472a7a', '#46307e', '#453781', '#433d84', '#414287', '#3f4889', '#3d4e8a', '#3a538b', '#38598c', '#355e8d', '#33638d', '#31688e', '#2e6d8e', '#2c718e', '#2a768e', '#297b8e', '#27808e', '#25848e', '#23898e', '#218e8d', '#20928c', '#1f978b', '#1e9c89', '#1fa188', '#21a585', '#24aa83', '#28ae80', '#2eb37c', '#35b779', '#3dbc74', '#46c06f', '#50c46a', '#5ac864', '#65cb5e', '#70cf57', '#7cd250', '#89d548', '#95d840', '#a2da37', '#b0dd2f', '#bddf26', '#cae11f', '#d8e219', '#e5e419', '#f1e51d', '#fde725']

try:
    EXP = PROCESSED_DIR.joinpath(sys.argv[1])
except IndexError:
    print("Specify run!")
    exit()


def sort_func(data):
    return data['iterations_to_gmp_convergence'][5]


data = []
for exp_run in [exp for exp in EXP.iterdir() if exp.is_file()]:
    data.append(read_write.load_json(exp_run, ''))
# data.sort(key=sort_func)


### AMPLITUDE ###
# for exp_run_idx, exp_run in enumerate(data):
#     xy = np.array(data[exp_run_idx]['xy'])
#     amps = []
#     for xy_idx in range(1, xy.shape[0]):
#         amps.append(0.5*(max(xy[:xy_idx, -1]) - min(xy[:xy_idx, -1])))
#     convergence = exp_run['iterations_to_gmp_convergence'][5]
#     plt.plot(np.arange(len(amps)), amps,
#              label=f'exp_{exp_run_idx+1}_{convergence}')#, c=COLORS[exp_run_idx])

# plt.xlabel('Number of iterations')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

### GMP ###
longest_run_duration = 0
convergence_times = []
tol = 0.1
for exp_run_idx, exp_run in enumerate(data):
    tolerances = np.array(data[exp_run_idx]['tolerance_levels'])
    gmp = np.array(data[exp_run_idx]['gmp'])
    longest_run_duration = max(longest_run_duration, gmp.shape[0])
    convergence_times.append(exp_run['iterations_to_gmp_convergence'][5])
    plt.plot(np.arange(gmp.shape[0]), gmp[:, -2],
             label=f'exp_{exp_run_idx+1}_{convergence_times[-1]}')
    plt.scatter(np.arange(gmp.shape[0]), gmp[:, -2],
             label=f'exp_{exp_run_idx+1}_{convergence_times[-1]}', s=10)
    if convergence_times[-1] is not None:
        plt.axvline(x=convergence_times[-1],
                    alpha=.5, color='red', linestyle='--')
plt.axhline(tol, alpha=.5, color='k', linestyle='dashed')
plt.axhline(-tol, alpha=.5, color='k', linestyle='dashed')


plt.ylim(-0.5, 2)
plt.legend()
plt.show()
