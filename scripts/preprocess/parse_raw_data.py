import numpy as np
import copy
import json
import os
import shutil
import preprocess
import sys
from pathlib import Path
# Add path to use read_write.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from read_write import load_yaml, load_json, save_json


# folder locations for raw and processed data
THESIS_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = THESIS_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = THESIS_DIR / 'data' / 'processed'
# CONFIG contains experiment names and truemin sources
CONFIG = load_yaml(THESIS_DIR.joinpath('scripts'), '/config.yaml')

verbose = False


def main():
    rm_tree(PROCESSED_DATA_DIR)     # removing existing directory if it exists
    PROCESSED_DATA_DIR.mkdir()
    parsed_data_dict = create_parsed_dict(RAW_DATA_DIR)
    all_experiments = list(parsed_data_dict.keys())

    for exp in all_experiments:
        exp_path = RAW_DATA_DIR.joinpath(exp)
        exp_batch = [x for x in exp_path.iterdir() if
                     x.is_dir() and 'exp' in str(x)]
        exp_batch.sort()
        for exp_run_idx, exp_run in enumerate(exp_batch):
            subruns = [x for x in exp_run.iterdir() if
                       x.is_dir() and 'subrun' in str(x)]
            if len(subruns) == 0:
                file_path = str(exp_run.joinpath('boss.out'))
                json_name = f'exp_{exp_run_idx+1}.json'
                PROCESSED_DATA_DIR.joinpath(exp).mkdir(parents=True,
                                                       exist_ok=True)
                json_path = str(PROCESSED_DATA_DIR.joinpath(exp, json_name))
                parse(file_path, exp, json_path)
            else:
                for subrun in subruns:
                    file_path = str(subrun.joinpath('boss.out'))
                    subrun_str = str(subrun).split('/')[-1]
                    json_name = f'exp_{exp_run_idx+1}_{subrun_str}.json'
                    PROCESSED_DATA_DIR.joinpath(exp).mkdir(parents=True,
                                                           exist_ok=True)
                    json_path = str(PROCESSED_DATA_DIR.joinpath(exp,
                                                                json_name))
                    parse(file_path, exp, json_path)

    # Once all the raw data is processed, substract the truemin
    # from the data. This needs to be done in another loop, since
    # the truemin comes from different sources

    # First, loop over the truemin source baseline experiments
    baselines = CONFIG['baselines']
    tolerances = CONFIG['tolerances']
    tl_experiments = CONFIG['TL_experiments']
    for exp in baselines:
        best_acqs = []
        sub_exp_paths = [
            x for x in PROCESSED_DATA_DIR.joinpath(baselines[exp]).iterdir()
                        ]
        sub_exp_paths.sort()
        truemin_precalculated = False
        truemin = None
        for sub_exp_path in sub_exp_paths:
            results = load_json('', sub_exp_path)
            if 'truemin' in results:
                truemin_precalculated = True
                break
            else:
                best_acqs.append(preprocess.get_best_acquisition(results))
        if truemin_precalculated is False:
            best_acqs = np.array(best_acqs)
            truemin = [best_acqs[np.argmin(best_acqs[:, -1]), :].tolist()]
            for sub_exp_path in sub_exp_paths:
                results = load_json('', sub_exp_path)
                results['truemin'] = truemin
                results = preprocess.preprocess(results, tolerances)
                save_json(results, sub_exp_path, '')

    # Secondly, loop over the other baseline experiments and add truemins
    for exp in baselines:
        sub_exp_paths = [
            x for x in PROCESSED_DATA_DIR.joinpath(exp).iterdir()]
        for sub_exp_path in sub_exp_paths:
            results = load_json('', sub_exp_path)
            if 'truemin' in results:
                # Truemin already calculated, go to next experiment
                break
            else:
                source_path = [
                    x for x in
                    PROCESSED_DATA_DIR.joinpath(baselines[exp]).iterdir()]
                # Only need the truemin from one truemin source
                # experiment, therefore access source_path[0]
                source = load_json('', source_path[0])
                results['truemin'] = source['truemin']
                results = preprocess.preprocess(results, tolerances)
                save_json(results, sub_exp_path, '')

    # Merge data from the baseline subruns
    for exp in baselines:
        if '_r' in exp:
            all_subrun_paths = sorted(
                [path for path in
                 PROCESSED_DATA_DIR.joinpath(exp).iterdir()])
            for sub_exp in parsed_data_dict[exp]:
                subrun_paths = [path for path in all_subrun_paths if
                                sub_exp in str(path)]
                merge_subrun_data(subrun_paths, sub_exp)
            subrun_dir = PROCESSED_DATA_DIR.joinpath(exp + '/subrun_files')
            subrun_dir.mkdir()
            for subrun in all_subrun_paths:
                if 'subrun' in str(subrun):
                    shutil.move(os.path.join(subrun_dir.parent, subrun),
                                subrun_dir)

    # BUG : Adjust this for the subrun structure, once data is available
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
                data = preprocess.preprocess(data, tolerances,
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
                    data = preprocess.preprocess(data, tolerances)
                    save_json(data, str(PROCESSED_DATA_DIR) + f'/{exp}',
                              f'/{filename}.json')

    # Merge data from the transfer learning subruns
    # TODO : Check if this works as intended, once data is available
    for exp in tl_experiments:
        if '_r' in exp:
            all_subrun_paths = sorted(
                [path for path in
                 PROCESSED_DATA_DIR.joinpath(exp).iterdir()])
            for sub_exp in parsed_data_dict[exp]:
                subrun_paths = [path for path in all_subrun_paths if
                                sub_exp in str(path)]
                merge_subrun_data(subrun_paths, sub_exp)
            subrun_dir = PROCESSED_DATA_DIR.joinpath(exp + '/subrun_files')
            subrun_dir.mkdir()
            for subrun in all_subrun_paths:
                if 'subrun' in str(subrun):
                    shutil.move(os.path.join(subrun_dir.parent, subrun),
                                subrun_dir)


def rm_tree(pth: Path):
    try:
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()
    except FileNotFoundError:
        return


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
    mt_keys = [key for key in data_dict if 'MT' in key]
    for key in mt_keys:
        del data_dict[key]
    save_json(data_dict, PROCESSED_DATA_DIR, '/parsed_dict.json')
    return data_dict


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
        if verbose:
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
                results['run_completed'] = [True]

    results['tasks'] = len(np.unique(np.array(xy)[:, -2]))
    if results['tasks'] not in [1, 2, 3]:
        results['tasks'] = 1
        results['dim'] = len(xy[0])-1
    else:
        # Hardcoded fix for 2UHFbasic0, where exp_6 has only two xy points:
        if len(np.array(xy)) == 2:
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

    if len(results['initpts']) == 1:
        # add 0, since no secondary task initpts used
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


def merge_subrun_data(subrun_file_paths, exp_idx):
    """Parses the subruns json files, cleans and merges
    the statistics. Saves the merged statistics as a single .json file.

    Args:
        subrun_file_paths (list): ordered list with paths to subruns
        exp_idx (str): Experiment to merge (e.g. exp_1)
    """
    # TODO : TL experiments have additional keys (e.g. 'B')
    # TODO : For the experiments that have been
    # unfinished ('run_completed' is False),
    # consider different stacking methods for different parameters
    # --> Here, it might be sufficient already to do
    #  if run_completed: continue
    # for some parameters
    to_copy_from_first_subrun = ['header', 'truemin',
                                 'thetainit', 'thetapriorparam', 'name',
                                 'bounds', 'dim', 'tasks', 'kernel',
                                 'num_tasks', 'tolerance_levels',
                                 'yrange', 'initpts']
    to_copy_from_last_subrun = ['xy']
    to_stack = ['GP_hyperparam', 'acq_times', 'best_acq', 'gmp_convergence',
                'run_completed']
    to_stack_and_clean = ['gmp', 'iterpts', 'iter_times',
                          'iterations_to_gmp_convergence', 'model_time',
                          'observations_to_gmp_convergence', 'total_time',
                          'totaltime_to_gmp_convergence']

    all_arguments = to_copy_from_first_subrun + to_copy_from_last_subrun + \
                    to_stack + to_stack_and_clean

    merged_results = dict.fromkeys([key for key in all_arguments])
    subrun_results = []

    # First part: Just copying (or stacking) the data
    for subrun_path in subrun_file_paths:
        subrun_results.append(load_json(subrun_path, ''))

    for key in to_copy_from_first_subrun:
        merged_results[key] = subrun_results[0][key]

    for key in to_copy_from_last_subrun:
        merged_results[key] = subrun_results[-1][key]

    for key in to_stack:
        merged_results[key] = []
        for subrun in subrun_results:
            values = subrun[key]
            for value in values:
                merged_results[key].append(value)

    # if any(np.bitwise_not(np.array(merged_results['run_completed']))):
    #     merged_results['unfinished_runs'] = np.argwhere(
    #         np.array(merged_results['run_completed']) == False)
    #     merged_results['unfinished_runs'] = \
    #         merged_results['unfinished_runs'][0].tolist()

    # Second part : Stack and clean the data, dependent on the key
    for key in to_stack_and_clean:
        merged_results[key] = []
        if key == 'gmp':
            for idx, subrun in enumerate(subrun_results):
                if idx != len(subrun_results) - 1 and \
                   subrun['run_completed'][0] is False:
                    continue
                values = subrun[key]
                if idx != 0:
                    values = values[1:] # First gmp from next subrun should
                                        # be ignored, this occurs twice
                for value in values:
                    merged_results[key].append(value)
        if key == 'iterpts':
            merged_results[key].append(
                len(subrun_results) * subrun_results[0][key][0])
        if key == 'iter_times':
            iterpts = subrun_results[0]['iterpts'][0]
            initpts = subrun_results[0]['initpts'][0]
            for idx, subrun in enumerate(subrun_results):
                values = subrun[key]
                start = (idx > 0)*initpts + idx*iterpts
                for value in values[start:]:
                    merged_results[key].append(value)
        if key == 'iterations_to_gmp_convergence':
            initial_initpts = subrun_results[0]['initpts'][0] + \
                subrun_results[0]['initpts'][1]
            for subrun_idx, subrun in enumerate(subrun_results):
                subrun_initpts = subrun['initpts'][0] + \
                    subrun['initpts'][1]
                values = subrun[key]
                if subrun_idx == 0:
                    merged_results[key] = copy.deepcopy(values)
                    continue
                for value_idx, value in enumerate(values):
                    if value is None:
                        sublist_length = len(merged_results[key][value_idx:])
                        merged_results[key][value_idx:] = sublist_length*[None]
                        break
                    elif value == 0:
                        continue
                    else:
                        merged_results[key][value_idx] = subrun_initpts + \
                            value - initial_initpts
        if key == 'totaltime_to_gmp_convergence':
            current_run_time = 0
            #previous_values, previous_run_completed = None, None
            for subrun_idx, subrun in enumerate(subrun_results):
                if subrun_idx == 0:
                    merged_results[key] = copy.deepcopy(subrun[key])
                    previous_values = copy.deepcopy(
                        subrun['iterations_to_gmp_convergence'])
                    previous_run_completed = subrun['run_completed'][0]
                    current_run_time += subrun['total_time'][-1]
                else:
                    values = subrun[key]
                    BO_iter_values = subrun['iterations_to_gmp_convergence']
                    for value_idx, value in enumerate(BO_iter_values):
                        if BO_iter_values[value_idx] is None:
                            N_sublist = len(merged_results[key][value_idx:])
                            merged_results[key][value_idx:] = N_sublist*[None]
                            current_run_time += subrun['total_time'][-1]
                            break
                        if BO_iter_values[value_idx] == 0:
                            if value_idx == len(BO_iter_values):
                                current_run_time += subrun['total_time'][-1]
                            else:
                                continue
                        else:
                            if previous_run_completed:
                                merged_results[key][value_idx] = \
                                    values[value_idx] + current_run_time
                            else:
                                # Only update, if in the continued run a new
                                # convergence tolerance was reached
                                prev_value = previous_values[value_idx]
                                if prev_value is None and value is not None:
                                    merged_results[key][value_idx] = \
                                        values[value_idx] + current_run_time
                                elif value - prev_value != 0:
                                    merged_results[key][value_idx] = \
                                        values[value_idx] + current_run_time
                                else:
                                    continue
                    previous_values = copy.deepcopy(BO_iter_values)
                    previous_run_completed = subrun['run_completed'][0]
        # TODO : Implement, this is not working yet (but also not used)
        if key == 'total_time':
            time_shift = 0
            for subrun in subrun_results:
                values = subrun[key]
                for value_idx, value in enumerate(values):
                    if value_idx < len(merged_results[key]):
                        continue
                    else:
                        merged_results[key].append(value + time_shift)
                time_shift += subrun['total_time'][-1]
        if key == 'observations_to_gmp_convergence':
            # This is not used
            pass  # TODO
        if key == 'model_time':
            # This is not used
            pass  # TODO

    json_name = exp_idx
    json_path = str(subrun_file_paths[0].parent) + '/'
    with open(os.path.expanduser(f'{json_path}{json_name}.json'), 'w') \
        as output_file:
            if verbose:
                print(f'Writing to file {json_path}{json_name}.json ...')
            json.dump(merged_results, output_file, indent=4)


if __name__ == '__main__':
    main()
