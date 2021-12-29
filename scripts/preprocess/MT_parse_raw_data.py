import numpy as np
import json
import os
import MT_preprocess
import sys
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json

THESIS_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG = load_yaml(THESIS_DIR.joinpath('scripts'), '/config.yaml')

RAW_DATA_DIR = THESIS_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = THESIS_DIR / 'data' / 'processed'


def main():

    # Parsing
    # exp = 'MT_a3b7'
    exp = 'MT_b3b1'
    exp_path = RAW_DATA_DIR.joinpath(exp)
    exp_batch = [x for x in exp_path.iterdir() if x.is_dir() and
                'exp' in str(x)]
    exp_batch.sort()
    for exp_idx, exp_run in enumerate(exp_batch):
        # parsing output file
        file_out = str(exp_run.joinpath('boss.out'))
        json_name = f'exp_{exp_idx+1}.json'
        PROCESSED_DATA_DIR.joinpath(exp).mkdir(parents=True, exist_ok=True)
        json_path = str(PROCESSED_DATA_DIR.joinpath(exp, json_name))

        file_rst = str(exp_run.joinpath('boss.rst'))
        file_paths = {'out': file_out, 'rst': file_rst}
        parse_outfile(file_paths, exp, json_path)

    # Preprocessing
    parsed_data_dict = create_parsed_dict(RAW_DATA_DIR)
    tl_experiments = CONFIG['TL_experiments']
    tolerances = CONFIG['tolerances']
    keys_to_remove = [key for key in tl_experiments if 'MT' not in key]
    for key in keys_to_remove:
        del tl_experiments[key]

    for exp in tl_experiments:
        truemin, init_times = [], []
        # Get data from all used baselines for initialization
        for i in range(len(tl_experiments[exp])):
            init_time = []
            baseline_exp = tl_experiments[exp][i][0]
            baseline_init_strategy = tl_experiments[exp][i][1]
            baseline_file = parsed_data_dict[baseline_exp][0]
            data = load_json(
                str(PROCESSED_DATA_DIR) +
                f'/{baseline_exp}/', f'{baseline_file}.json')
            truemin.append(data['truemin'][0])

            if baseline_init_strategy == 'self':
                init_time = None    # This is 'BO random', not used anymore
            elif baseline_init_strategy == 'random':
                for baseline_file in parsed_data_dict[baseline_exp]:
                    data = load_json(
                        str(PROCESSED_DATA_DIR) +
                        f'/{baseline_exp}/', f'{baseline_file}.json')
                    additional_time = data['acq_times'].copy()
                    for i in range(len(data['acq_times'])):
                        additional_time[i] += \
                            sum(np.array(data['acq_times'])[:i])
                    init_time.append(additional_time)
            elif baseline_init_strategy == 'inorder':
                for baseline_file in parsed_data_dict[baseline_exp]:
                    data = load_json(
                        str(PROCESSED_DATA_DIR) +
                        f'/{baseline_exp}/', f'{baseline_file}.json')
                    init_time.append(data['total_time'].copy())
            else:
                raise ValueError("Unknown initialization strategy")
            init_times.append(init_time)

        for tl_exp_idx, _ in enumerate(parsed_data_dict[exp]):
            initial_data_cost = []
            for init_time in init_times:
                if init_time is None:
                    initial_data_cost.append(None)
                else:
                    N_baselines = len(init_time)
                    initial_data_cost.append(init_time[(tl_exp_idx
                                                        % N_baselines)])
            filename = parsed_data_dict[exp][tl_exp_idx]
            # TODO : Check if this works as intended
            # preprocess.preprocess on subrun_1 WITH initial datacost,
            # on the other runs WITHOUT initial datacost
            if '_r' not in exp:
                data = load_json(str(PROCESSED_DATA_DIR) +
                                 f'/{exp}', f'/{filename}.json')
                data['truemin'] = truemin
                data = MT_preprocess.preprocess(data, tolerances,
                                             initial_data_cost)
                save_json(data, str(PROCESSED_DATA_DIR) + f'/{exp}',
                          f'/{filename}.json')
            # TODO : Go through the following in detail again! It's
            # likely that this isn't working as intended yet
            else:
                data_paths = [
                    path for path in PROCESSED_DATA_DIR.joinpath(exp).iterdir()
                    if filename in str(path)]
                data_paths.sort()
                for data_path in data_paths:
                    filename = str(data_path).split('/')[-1].split('.')[0]
                    data = load_json(str(data_path), '')
                    data['truemin'] = truemin
                    data = MT_preprocess.preprocess(data, tolerances)
                    save_json(data, str(PROCESSED_DATA_DIR) + f'/{exp}',
                              f'/{filename}.json')



def read_boss_rst(path, file_name, results):
    path = os.path.expanduser(path)
    results['task_index'] = []
    task_index = []
    with open(''.join((path, file_name)), 'r') as file:
        lines = file.readlines()
        start_parsing = False
        for line_idx in range(len(lines)):
            line = lines[line_idx]
            if 'initpts' in line and start_parsing is False:
                results['initpts'] = parse_values(line)
            elif 'RESULTS' in line:
                start_parsing = True
                continue
            if start_parsing:
                task_index.append(int(
                    parse_values(line, typecast=float, idx=0)[results['dim']]))
    results['task_index'] = task_index

    # List of keys where the task index has to be added manually
    task_index_to_be_added = ['best_acq', 'gmp', 'xy', 'task_index']
    for key in task_index_to_be_added:
        results[key] = np.array(results[key])

    # Merging ...
    start_idx = sum(results['initpts']) - 1
    results['xy'] = np.insert(results['xy'], results['dim'],
                              results['task_index'], axis=1)

    # The best_acq and gmp lines in the outfile are always for the actual
    # objective function, therefore they all have taskindex 0
    results['best_acq'] = np.insert(results['best_acq'], results['dim'],
                                    0.0*results['task_index'][start_idx:],
                                    axis=1)
    results['gmp'] = np.insert(results['gmp'], results['dim'],
                               0.0*results['task_index'][start_idx:], axis=1)
    # Transforming back to list, so can be saved as json
    for key in task_index_to_be_added:
        results[key] = results[key].tolist()

    return results


def read_boss_output(path, file_name, exp_name):
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
               'inittype': None,
               'bounds': None,
               'num_tasks': 1,
               'acq_times': None,
               'best_acq': None,
               'gmp': None,
               'gmp_convergence': None,
               'GP_hyperparam': None,
               'iter_times': None,
               'total_time': None,
               'run_completed': [False],
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
            elif 'initpts' in line:
                # This doesn't work yet with the MT output file
                results['initpts'] = parse_values(line, cut_idx=-2)
                results['iterpts'] = parse_values(line, idx=3)
            elif 'inittype' in line:
                results['inittype'] = parse_values(line, typecast=str)
                results['num_tasks'] = len(results['inittype'])
            elif 'bounds' in line and results['bounds'] is None:
                tmp = ' '.join(parse_values(line, typecast=str, idx=1))
                results['bounds'] = parse_values(tmp,
                    typecast=str, sep=';', idx=0)
            # elif 'kernel' in line:
            #     results['kernel'] = parse_values(line, typecast=str, idx=1)
            elif 'kerntype' in line:
                results['kernel'] = parse_values(line, typecast=str)
            elif 'yrange' in line:
                results['yrange'] = parse_values(line, typecast=str)
            elif 'thetainit' in line:
                results['thetainit'] = parse_values(line, typecast=str)
            elif 'thetapriorpar' in line:
                tmp = ' '.join(parse_values(line, typecast=str))
                results['thetapriorpar'] = parse_values(tmp, typecast=str,
                    sep=';', idx=0)
            elif '|| Bayesian optimization completed' in line:
                results['run_completed'] = [True]

    results['xy'] = xy
    results['dim'] = len(xy[0]) - 1
    results['acq_times'] = acq_times
    results['best_acq'] = best_acq
    results['gmp'] = global_min_prediction
    results['gmp_convergence'] = global_min_prediction_convergence
    results['GP_hyperparam'] = gp_hyperparam
    results['iter_times'] = iter_times
    results['total_time'] = total_time

    return results


def parse_values(line, typecast=int, sep=None, idx=1, cut_idx=None):
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
    if cut_idx is not None:
        return [typecast(val.strip(sep))
                for val in line.split(sep)[idx:cut_idx]]
    else:
        return [typecast(val.strip(sep)) for val in line.split(sep)[idx:]]


def parse_outfile(input_file_path, exp_name, output_file_path):
    """Parses the boss.out input file and saves the data as dict to .json file.

    Args:
        input_file_path (string): boss.out file path
        exp_name (string): Name of the descriptive experiment
        output_file_path (string): .json file path.
    """
    output_file = output_file_path.split('.json')[0]
    save_to_json('', input_file_path, exp_name, '', output_file)


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

    if isinstance(file_name, dict):
        results = read_boss_output(path,
                                                  file_name['out'], exp_name)
        results = read_boss_rst(path, file_name['rst'], results)
    else:
        results = read_boss_output(path, file_name, exp_name)

    # expanduser expands an initial path component (~) in the given
    # path to the users home dir
    with open(os.path.expanduser(f'{json_path}{json_name}.json'), 'w') \
         as output_file:
        json.dump(results, output_file, indent=4)


def create_parsed_dict(data_folder):
    """Create a dict, listing all the runs for an experiment.

    Args:
        data_folder (str): Path to raw data.

    Returns:
        dict: Dict, listing the runs per experiments.
    """
    data_dict = dict()
    for exp in data_folder.iterdir():
        if exp.is_dir() and 'misc' not in str(exp):
            exp_runs = []
            for exp_run in exp.iterdir():
                if exp_run.is_dir() and 'exp' in str(exp_run):
                    exp_run = str(exp_run).split('/')[-1]
                    exp_runs.append(exp_run)
            exp_name = str(exp).split(sep='/')[-1]
            exp_runs = sorted(exp_runs, key=lambda run:
                              int(run.split(sep='_')[-1]))
            data_dict[exp_name] = exp_runs
    save_json(data_dict, PROCESSED_DATA_DIR, '/parsed_dict.json')
    return data_dict


if __name__ == '__main__':
    main()
