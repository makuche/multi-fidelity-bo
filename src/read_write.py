import json
import yaml
import pandas as pd


def load_experiments(experiments):
    """Given a list of experiment paths, load the data and return a list of
    the loaded experiments.

    Parameters
    ----------
    experiments : list
        List of Path objects to experiments

    Returns
    -------
    list
        Loaded experiments.
    """
    experiments_data = []
    for experiment in experiments:
        exp_data = []
        exp_runs = [exp for exp in experiment.iterdir() if exp.is_file()]
        # The following sorts the experiments by the experiment number
        # (e.g. exp_1, exp_2, ...)
        exp_runs.sort(
            key=lambda string: int(str(string).split('_')[-1].split('.')[0]))
        for exp in exp_runs:
            exp_data.append(load_json('', exp))
        experiments_data.append(exp_data)
    return experiments_data


def load_statistics_to_dataframe(baseline_experiments, tl_experiments,
                                 num_exp=None):
    columns = [key for key in tl_experiments[0][0].keys()]
    df = pd.DataFrame(columns=columns)


    for baseline_experiment in baseline_experiments:
        for baseline_run in baseline_experiment[:num_exp]:
            entry = correct_type_for_dataframe(baseline_run)
            df = pd.concat([df, entry], axis=0)
    for tl_experiment in tl_experiments:
        for tl_run in tl_experiment[:num_exp]:
            entry = correct_type_for_dataframe(tl_run)
            df = pd.concat([df, entry], axis=0)
    return df


def correct_type_for_dataframe(result):
    """Utility function to format results (saved in a dictionary) to
    a dataframe.

    This causes problems, if the entries in the dict have different lengthts.
    A quick fix solution, is to put all entries into a list with the
    same length.

    Parameters
    ----------
    results : dict
        Dictionary containing parsed results.

    Returns
    -------
    Dataframe
        Returns dataframe that can be concatenated to a dataframe.
    """
    entries_to_lists, lists_with_length_one = {}, {}
    for key in result:
        if not isinstance(result[key], list):
            entries_to_lists[key] = [result[key]]
        else:
            entries_to_lists[key] = result[key]
    for key in entries_to_lists:
        if len(entries_to_lists[key]) == 1:
            lists_with_length_one[key] = entries_to_lists[key]
        else:
            lists_with_length_one[key] = [entries_to_lists[key]]
    return pd.DataFrame(lists_with_length_one)


def load_json(path, filename):
    with open(f'{path}{filename}', 'r') as f:
        return json.load(f)


def save_json(data, path, filename):
    with open(f'{path}{filename}', 'w') as f:
        json.dump(data, f, indent=4)


def load_yaml(path, filename):
    with open(f'{path}{filename}', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
