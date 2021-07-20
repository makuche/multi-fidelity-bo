import json
import yaml


def load_json(path, filename):
    """[summary]

    Args:
        path ([type]): [description]
        filename ([type]): [description]

    Raises:
        FileNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data
    raise FileNotFoundError(
        f'{path}{filename} could not be loaded with json.load')


def save_json(data, path, filename):
    """[summary]

    Args:
        data ([type]): [description]
        path ([type]): [description]
        filename ([type]): [description]
    """
    with open(f'{path}{filename}', 'w') as f:
        json.dump(data, f, indent=4)


def load_yaml(path, filename):
    """[summary]

    Args:
        path ([type]): [description]
        filename ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(f'{path}{filename}', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

