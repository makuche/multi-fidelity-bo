import numpy as np


def get_best_acquisition(data):
    """Returns coordinates x and f(x) for lowest observed acquisition.

    Args:
        data (dict): Parsed data from boss.out
    """
    return np.array(data['best_acq'])[-1, :]


def substract_y_offset(data):
    """Adds the offsets (= lowest observed acquisition) to the energy values.

    Args:
        data (dict): Parsed data from boss.out
    """
    y_offset = np.array(data['truemin'])[:, -1]
    for value in data['gmp']:
        value[-2] -= y_offset[0]
    for value in data['best_acq']:
        value[-1] -= y_offset[0]

    N_sources = len(y_offset)
    if N_sources == 1:                          # Baseline runs
        for value in data['xy']:
            value[-1] -= y_offset[0]
    elif N_sources > 1:                         # TL runs
        for value in data['xy']:
            for i in range(N_sources):
                if value[-2] == i:
                    value[-1] -= y_offset[i]


def add_init_acq_times(data, init_data_cost):
    """Adds computational cost for the initial acquisitions, used for
    Transfer Learning.

    data['initpts'] contains a list with the
    number of initial points for each fidelity. E.g. [2, 0].

    Args:
        data (dict): Parsed data from boss.out
        init_data_cost (list): List containing the additional acquisition
        times from the initialization data.
    """
    accounted_initpts = 0
    for cost, initpts in zip(init_data_cost, data['initpts']):
        # init strategy 'self'
        if cost is None:
            pass
        # init data is taken from baseline
        else:
            begin = accounted_initpts
            if accounted_initpts != 0:
                begin -= 1
            end = begin + initpts
            for i in range(begin, end):
                data['total_time'][i] += cost[i-begin]
            for i in range(end, len(data['total_time'])):
                data['total_time'][i] += cost[initpts-1]
        accounted_initpts += initpts


def calculate_convergence_times(data, idx, measure='gmp',
                                check_also_xhat=False):
    """Calculates the convergence points/times of a quantity for given
    tolerances.

    Args:
        data (dict): Cotanins the parsed data.
        idx (int): Dimension where data is located (has to be taken
        into account for MT runs).
        measure (str, optional): Quantity to calculate convergence measures
        for. Defaults to 'gmp'.
        check_also_xhat (bool, default=False): Consider also xhat (the
        search space location) of the GMP as a convergence measure.
    """
    if data[measure] == []:
        return          # This is for interrupted sobol runs (hardcoded fix)
    values = np.atleast_2d(data[measure])[:, idx][::-1]     # note reversion
    data[f'iterations_to_{measure}_convergence'] = []       # BO iterations
    data[f'totaltime_to_{measure}_convergence'] = []        # total runtime
    # TODO : Check the following, if correctly calculated or if this
    # is used at all -> If not, delete it
    data[f'observations_to_{measure}_convergence'] = []     # BO + init points

    if not check_also_xhat:
        for tolerance in data['tolerance_levels']:
            i = 0
            for value in values:
                if abs(value) > tolerance:
                    break
                i += 1
            if i == 0:
                iterations = None
                totaltime = None
                observations = None
            else:
                iterations = len(values) - i
                totaltime = data['total_time'][-i]
                observations = len(data['xy']) - i
            data[f'iterations_to_{measure}_convergence'].append(iterations)
            data[f'totaltime_to_{measure}_convergence'].append(totaltime)
            data[f'observations_to_{measure}_convergence'].append(observations)
    else:
        def point_within_hypercube(xhat_coords, cube_center, sidelength=20):
            """
            Checks if a point is within a N dimensional
            hypercube with certain sidelength. For each dimension d, it is
            checked that |xhat[d] - cube_center[d]| < sidelength.

            This can be used to check also the xhat from the predicted
            global minimum, to see if the global structure search
            succeeded.
            """
            dimension_convered = []
            for coord, cube_coord in zip(xhat_coords, cube_center):
                dimension_convered.append(
                    np.abs(coord - cube_coord) < sidelength)
            return np.all(dimension_convered)
        predict_xhats = np.atleast_2d(data[measure])[:, 0:idx][::-1]
        true_xhat = np.array(data['truemin'])[0][:-1]
        predict_yhats = np.atleast_2d(data[measure])[:, idx][::-1]
        for tolerance in data['tolerance_levels']:
            i = 0
            for predict_yhat, predict_xhat in zip(predict_yhats, predict_xhats):
                if abs(predict_yhat) > tolerance or not\
                   point_within_hypercube(predict_xhat, true_xhat):
                    break
                i += 1
            if i == 0:
                iterations = None
                totaltime = None
                observations = None
            else:
                iterations = len(values) - i
                totaltime = data['total_time'][-i]
                observations = len(data['xy']) - i
            data[f'iterations_to_{measure}_convergence'].append(iterations)
            data[f'totaltime_to_{measure}_convergence'].append(totaltime)
            data[f'observations_to_{measure}_convergence'].append(observations)


def calculate_B(data):
    dim = data['dim']
    if dim == len(data['xy'][0])-1:
        data['B'] = None
    else:
        data['B'] = []
        tasks = data['tasks']
        for params in data['GP_hyperparam']:
            W = np.array(params[dim:-tasks]).reshape((-1, tasks))
            Kappa = np.diag(params[-tasks:])
            B = W.dot(W.T) + Kappa
            data['B'].append([b for b in B.flatten()])


def preprocess(data, tolerance_levels=[0], init_data_cost=None):
    """Adds time taken for initialization data (acquisition time). #
    Calculates model time.
    Rescales output so that best acquisition of the baseline experiment is 0.
    Calculates convergence.


    Args:
        data (dict): Dict containing the data of boss.out
        tolerance_levels (list, optional): List of tolerance levels.
        Defaults to [0].
        init_data_cost (float, optional): Cost of the initial data.
        Defaults to None.
    """
    # Add times for the GPR model
    data['model_time'] = [iter_time-acq_time for iter_time, acq_time in
                          zip(data['iter_times'][::-1],
                              data['acq_times'][::-1])]
    data['model_time'] = data['model_time'][::-1]

    # Adds (possible) extra cost for initialization data
    if init_data_cost is not None:
        add_init_acq_times(data, init_data_cost)

    # Add offset to the data and add convergence times
    substract_y_offset(data)
    data['tolerance_levels'] = tolerance_levels
    calculate_convergence_times(data, idx=-2, check_also_xhat=False)

    calculate_B(data)

    return data
