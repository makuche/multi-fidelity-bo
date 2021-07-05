import json
import yaml

def load_json(path, filename):
    """
    load json file
    """
    with open(f'{path}{filename}', 'r') as f:
        data = json.load(f)
        return data
    raise FileNotFoundError(f'{path}{filename} could not be loaded with json.load')

def save_json(data, path, filename):
    """
    save json file
    """
    with open(f'{path}{filename}', 'w') as f:
        json.dump(data, f)

def load_yaml(path, filename):
    """
    load yaml file
    """
    with open(f'{path}{filename}', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def textabline(row):
    line = ' & '.join(row)
    line = f'{line} \\\\ \n'
    return line
def write_table_tex(table, filepath, colnames = None, rownames = None):
    """
    write latex table
    """
    lines = []
    with open(filepath, 'w') as f:
        if colnames is not None:
            lines.append(textabline(colnames))
        if rownames is None:
            for row in table:
                row = ['{:.3f}'.format(val) for val in row]
                lines.append(textabline(row))
        else:
            for row, rowname in zip(table, rownames):
                row = ['{:.3f}'.format(val) for val in row]
                row.insert(0, rowname)
                line = textabline(row)
                lines.append(line)
        f.writelines(lines)
