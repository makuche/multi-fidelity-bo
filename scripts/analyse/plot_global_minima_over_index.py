import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import dirname, abspath, join as joinpath
from pathlib import Path

# Clunky os module usage (requires function nesting, has to be read from
# inside to the outside)
thesis_dir = dirname(dirname(dirname(abspath(__file__))))
processed_data_dir = joinpath(thesis_dir, 'data/processed')
print(processed_data_dir)

# versus

# Better ordered pathlib usage (allows us to chain methods and attributes on
# Path)
thesis_dir = Path(__file__).resolve().parent.parent
processed_data_dir = thesis_dir.joinpath('data/processed')
# Alternatively :
processed_data_dir = thesis_dir.joinpath('data').joinpath('processed')
exit()
# path_B1 = Path("../../data/processed/UHF/exp_1.json")
# print(path_B1.absolute())
# path_B2 = Path("../../data/processed/UHF_newbasis/exp_1.json")
PATHS = [path_B1, path_B2]
DATA = []

def main():
    for path in PATHS:
        with open(f'{path.absolute()}' 'r') as file:
            DATA.append(json.load(file))

if __name__ == '__main__':
    main()
