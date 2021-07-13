import numpy as np
import json
import os
import preprocess
import copy
from pathlib import Path

import read_write


# folder locations for raw and processed data
THESIS_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = THESIS_DIR.joinpath('data').joinpath('raw')
PROCESSED_DATA_DIR = THESIS_DIR.joinpath('data').joinpath('processed')
# CONFIG contains experiment names and truemin sources
CONFIG = read_write.load_yaml(
    THESIS_DIR.joinpath('scripts/preprocess/config'), '/preprocess.yaml')


def main():
    # TODO : Take this part later to another script
    for exp in CONFIG['baselines'].keys():
        exp_path = RAW_DATA_DIR.joinpath(exp)
        sub_exp_paths = [x for x in exp_path.iterdir() if
                         (x.is_dir() and 'job' not in str(x) and
                          'template' not in str(x))]
        sub_exp_paths.sort()
        for idx, sub_exp in enumerate(sub_exp_paths):
            # Casting the pathlib objects to str, so methods like .split()
            # can be used
            file_path = str(sub_exp.joinpath('boss.out'))
            str_ = f'exp_{idx+1}.json'
            PROCESSED_DATA_DIR.joinpath(exp).mkdir(parents=True, exist_ok=True)
            json_path = str(PROCESSED_DATA_DIR.joinpath(exp).joinpath(str_))
            parse(file_path, exp, json_path)

    # Once all the raw data is processed, substract the truemin
    # from the data. This needs to be done in another loop, since
    # the truemin comes from different sources

    # First, loop over the truemin source baseline experiments
    baselines = CONFIG['baselines']
    for exp in baselines:
        best_acqs = []
        sub_exp_paths = [
            x for x in PROCESSED_DATA_DIR.joinpath(baselines[exp]).iterdir()
                        ]
        sub_exp_paths.sort()
        truemin_precalculated = False
        truemin = None
        for sub_exp_path in sub_exp_paths:
            results = read_write.load_json('', sub_exp_path)
            if 'truemin' in results:
                truemin_precalculated = True
                break
            else:
                best_acqs.append(preprocess.get_best_acquisition(results))
        if truemin_precalculated is False:
            best_acqs = np.array(best_acqs)
            truemin = [best_acqs[np.argmin(best_acqs[:, -1]), :].tolist()]
            for sub_exp_path in sub_exp_paths:
                results = read_write.load_json('', sub_exp_path)
                results['truemin'] = truemin
                print("Saving to ...", str(sub_exp_path))
                results = preprocess.preprocess(results)
                read_write.save_json(results, sub_exp_path, '')

    # Secondly, loop over the other baseline experiments and 'attach'
    # the truemins from the truemin sources
    for exp in baselines:
        sub_exp_paths = [
            x for x in PROCESSED_DATA_DIR.joinpath(exp).iterdir()]
        for sub_exp_path in sub_exp_paths:
            results = read_write.load_json('', sub_exp_path)
            if 'truemin' in results:
                # Truemin already calculated, go to next experiment
                break
            else:
                source_path = [
                    x for x in
                    PROCESSED_DATA_DIR.joinpath(baselines[exp]).iterdir()]
                # Only need the truemin from one truemin source
                # experiment, therefore access source_path[0]
                source = read_write.load_json('', source_path[0])
                results['truemin'] = source['truemin']
                print("Saving to ...", str(sub_exp_path))
                results = preprocess.preprocess(results)
                read_write.save_json(results, sub_exp_path, '')


def parse_values(line, typecast=int, sep=None, idx=1):
    """Returns a list of parsed values from a line in the output file.

    Args:
        line (string): Line to parse values from.
        typecast (<type>, optional): Type to converse values to. Defaults
        to int.
        sep (string, optional): Separator of the values in the string.
        Defaults to None.
        idx (Starting index for list to parse from, optional): . Defaults to 1.

    Returns:
        list: List including all parsed values from the line string in
        'correct' format
    """
    # .strip removes spaces from the beginning and end of the string
    return [typecast(val.strip(sep)) for val in line.split(sep)[idx:]]


def save_to_json(path, file_name, exp_name, json_path=None, json_name=None):
    """Parse the results and save to json file.

    Args:
        path (string): Path to folder where boss.out file is.
        file_name (string): Name of boss.out file.
        exp_name (string): Name of the descriptive experiment.
        json_path (string, optional): Path of json file. Defaults to path.
        json_name (string, optional): Name of json file. Defaults to exp_name.
    """
    if json_path is None:
        json_path = path
    if json_name is None:
        json_name = exp_name

    results = read_and_preprocess_boss_output(path, file_name, exp_name)
    # expanduser expands an initial path component (~) in the given
    # path to the users home dir
    with open(os.path.expanduser(f'{json_path}{json_name}.json'), 'w') \
        as output_file:
        print(f'Writing to file {json_path}{json_name}.json ...')
        json.dump(results, output_file, indent=4)


def read_and_preprocess_boss_output(path, file_name, exp_name):
    """Reads boss.out file and returns a dict() with parsed values.

    Args:
        path (string): Path to folder where boss.out is.
        file_name (string): Name of boss.out file.
        exp_name (string): Name of descriptive experiment.
    """
    path = os.path.expanduser(path)
    results = {'name': exp_name,
               'initpts': None,
               'iterpts': None,
               'bounds': None,
               'num_tasks': 1,
               'obs': None,           # TODO : Check what that was
               'acq_times': None,
               'best_acq': None,
               'gmp': None,
               'gmp_convergence': None,
               'GP_hyperparam': None,
               'iter_times': None,
               'total_time': None,
               'run_completed': False,
               }
    xy = []
    acq_times = []
    best_acq = []
    global_min_prediction = []
    global_min_prediction_convergence = []
    gp_hyperparam = []
    iter_times = []
    total_time = []
    with open(''.join((path, file_name)), 'r') as file:
        lines = file.readlines()
        results['header'] = lines[0:100]
        for i in range(len(lines)):
            line = lines[i]
            if 'Data point added to dataset' in line:
                line = lines[i+1]
                xy.append(parse_values(line, typecast=float, idx=0))
            elif 'Best acquisition' in line:
                line = lines[i+1]
                best_acq.append(parse_values(line, typecast=float, idx=0))
            elif 'Global minimum prediction' in line:
                line = lines[i+1]
                global_min_prediction.append(parse_values(line, typecast=float,
                                                          idx=0))
            # TODO : Check what the following parses for (maybe something in
            # BOSS_MT)
            elif 'Global minimum convergence' in line:
                line = lines[i+1]
                global_min_prediction_convergence.append(
                    parse_values(line,
                                 typecast=float,
                                 idx=0)
                                 )
            elif 'GP model hyperparameters' in line:
                line = lines[i+1]
                gp_hyperparam.append(parse_values(line, typecast=float, idx=0))
            elif 'Iteration time [s]:' in line:
                # If line contains str and float types, casting to str and then
                # manually to float again has to be done
                iter_times.append(float(
                    parse_values(line, typecast=str, idx=3)[0]))
                # Here not needed because line only contains a float
                total_time.append(parse_values(line, typecast=float, idx=7)[0])
            elif 'Objective function evaluated' in line:
                acq_times.append(parse_values(line, typecast=float, idx=6)[0])
            elif 'initpts' in line and results['initpts'] is None:
                results['initpts'] = parse_values(line)
            elif 'iterpts' in line and results['iterpts'] is None:
                results['iterpts'] = parse_values(line)
            elif 'num_tasks' in line:
                results['num_tasks'] = parse_values(line)[0]
            elif 'bounds' in line and results['bounds'] is None:
                tmp = ' '.join(parse_values(line, typecast=str, idx=1))
                results['bounds'] = parse_values(tmp,
                    typecast=str, sep=';', idx=0)
            elif 'kernel' in line:
                results['kernel'] = parse_values(line, typecast=str, idx=1)
            elif 'yrange' in line:
                results['yrange'] = parse_values(line, typecast=str, idx=1)
            elif 'thetainit' in line:
                results['thetainit'] = parse_values(line, typecast=str, idx=1)
            elif 'thetapriorparam' in line:
                tmp = ' '.join(parse_values(line, typecast=str, idx=1))
                results['thetapriorparam'] = parse_values(tmp, typecast=str,
                    sep=';', idx=0)
            elif '|| Bayesian optimization completed' in line:
                results['run_completed'] = True

    # TODO : Check that (also with the new boss)
    results['tasks'] = len(np.unique(np.array(xy)[:, -2]))
    if results['tasks'] not in [1, 2, 3]:
        results['tasks'] = 1
        results['dim'] = len(xy[0])-1
    else:
        results['dim'] = len(xy[0])-2

    results['xy'] = xy
    results['acq_times'] = acq_times
    # These are the sobol runs that did not finish properly:
    if best_acq == []:
        acqs = np.array(results['xy'])
        best_acq.append(acqs[np.argmin(acqs[:, -1]), :].tolist())
    results['best_acq'] = best_acq
    results['gmp'] = global_min_prediction
    results['gmp_convergence'] = global_min_prediction_convergence
    results['GP_hyperparam'] = gp_hyperparam
    results['iter_times'] = iter_times
    results['total_time'] = total_time

    # 0 stands for the init points of the secondary task
    # (TODO : check if thats correct)
    if len(results['initpts']) == 1: # add 0 secondary initpts
        results['initpts'].append(0)

    return results


def parse(input_file_path, exp_name, output_file_path):
    """Parses the boss.out input file and saves the data as dict to .json file.

    Args:
        input_file_path (string): boss.out file path
        exp_name (string): Name of the descriptive experiment
        output_file_path (string): .json file path.
    """
    output_file = output_file_path.split('.json')[0]
    save_to_json('', input_file_path, exp_name, '', output_file)


if __name__ == '__main__':
    main()
