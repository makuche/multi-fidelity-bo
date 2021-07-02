import numpy as np
import json
import os 
import sys
import yaml

def get_best_acquisition(data):
    """Returns coordinates x and f(x) for lowest observed acquisition.

    Args:
        data (dict): Parsed data from boss.out
    """
    return np.array(data['best_acq'][-1,:])

def substract_y_offset(data):
    """Adds the offset (= lowest observed acquisition) to the energy values.

    Args:
        data (dict): Parsed data from boss.out
    """
    y_offset = np.array(data['truemin'])[:,-1]

    for value in data['gmp']:
        value[-2] -= y_offset[0]
    for value in data['best_acq']:
        value[-1] -= y_offset[0]

    # Multi task BOSS, TODO : Check again once MT is set up
    for i in range(len(y_offset)):
        for value in data['xy']:
            if i == 0:
                value[-1] -= y_offset[i]
            elif len(y_offset)  > 0 and value[-2] == i:
                value[-1] -= y_offset[i]

def preprocess(data, tolerance_levels = [0], initial_data_cost = None):
    """Adds time taken for initialization data (acquisition time). Calculates model time.
    Rescales output so that best acquisition of the baseline experiment is 0.
    Calculates convergence.


    Args:
        data (dict): Dict containing the data of boss.out
        tolerance_levels (list, optional): List of tolerance levels. Defaults to [0].
        initial_data_cost (float, optional): Cost of the initial data. Defaults to None.
    """
    # Add times for the GPR model
    data['model_time'] = [iter_time-acq_time for iter_time, acq_time in \
        zip(data['iter_times'], data['acq_times'])]

    # Adds (possible) extra cost for initialization data
    if initial_data_cost is not None:
        pass # TODO : Add once MT is set up

    # Add offset to the data
    substract_y_offset(data)

    # TODO : Add calculate_convergence function
    # TODO : Add calculate B
    return data