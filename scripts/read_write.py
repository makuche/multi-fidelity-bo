import json
import yaml


def load_json(path, filename):
    with open(f'{path}{filename}', 'r') as f:
        return json.load(f)


def save_json(data, path, filename):
    with open(f'{path}{filename}', 'w') as f:
        json.dump(data, f, indent=4)


def load_yaml(path, filename):
    with open(f'{path}{filename}', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
