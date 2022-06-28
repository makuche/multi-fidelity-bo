from os import remove
import numpy as np
import matplotlib.pyplot as plt
import click
import shutil

from pathlib import Path

# TODO Put here what is currently out_puhti
DATA_DIR = Path(__file__).resolve().parent.parent  / 'data'
DATA_DIR = DATA_DIR / 'merge_outputs/'


@click.command()
@click.option('--setup', required=True, help='Experimental setup')
@click.option('--path', required=True, default=DATA_DIR,
              help='Experimental setup')
@click.option('--all', is_flag=True, default=False)

def main(setup, path):
    data_dir = Path(path) / setup
    print(path)
    exit()
    experiments = [exp for exp in data_dir.iterdir() if (exp.is_dir()
                   and 'template' not in exp.name)]
    experiments.sort()
    for exp_path in experiments:
        outfile_text = merge_subrun_output_files(exp_path)
        # Save new output file
        with open(exp_path / 'boss.out', 'w') as f:
            f.write(outfile_text)

        # Copy last rst file
        copy_last_rst_file(exp_path)
#        shutil.copy(exp_path / 'boss.rst', exp_path)


def merge_subrun_output_files(data_dir):
    subruns = [dir_ for dir_ in data_dir.iterdir() if dir_.is_dir()]
    subruns.sort()

    # Read data from first subrun:
    # Copy everything until the last finished iteration
    lines = get_textlines_from_file(subruns[0] / 'boss.out')
    lines_cleaned = copy_until_last_iteration(lines)
    text = ''.join(lines_cleaned)

    # Second to last subrun:
    # Start copying from the first iteration until the last finished iteration
    for subrun in subruns[1:]:
        lines = get_textlines_from_file(subrun / 'boss.out')
        if lines is None:
            continue
        lines_cleaned = copy_from_first_until_last_iteration(lines)
        if lines_cleaned is not None:
            text_next_subrun = ''.join(lines_cleaned)
            text = text + text_next_subrun


#   TODO : Skipping the tail for now, maybe add later
    # Get the tail from last BOSS subrun
#    lines = get_textlines_from_file(subruns[-1] / 'boss.out')
#    tail = get_tail_from_last_iteration(lines)
#    if tail is not None:
#        text = text + ''.join(tail)


    # Correct the 'total time' information in output file, as each subrun
    # starts the timer from 0 again
    timings, indices = get_timings(text.splitlines())
    timings = calculate_correct_timings(timings)
    lines = correct_timings(text.splitlines(), timings)
    text = ''.join(lines)
    return text


def copy_last_rst_file(exp_path):
    subruns = [dir_ for dir_ in exp_path.iterdir() if dir_.is_dir()]
    subruns.sort()
    last_subrun = subruns[-1]
    shutil.copy(last_subrun / 'boss.rst', exp_path)


def get_textlines_from_file(file_path):
    try:
        with open(file_path) as f:
            return f.readlines()
    except FileNotFoundError:
        return None


def copy_until_last_iteration(lines):
    reversed_lines = lines[::-1]
    for line_idx, line in enumerate(reversed_lines):
        if "Iteration time" in line:
            start_idx = line_idx
            break
    return lines[:-start_idx]


def copy_from_first_until_last_iteration(lines):
    reversed_lines = lines[::-1]
    start_idx = None
    for line_idx, line in enumerate(reversed_lines):
        if "Objective function evaluated" in line:
            start_idx = line_idx - 10
            break
    if start_idx is None:
        return None
    lines = lines[:-start_idx]
    for line_idx, line in enumerate(lines):
        if "Objective function evaluated" in line:
            start_idx = line_idx - 4
            break
    return lines[start_idx:]


def get_tail_from_last_iteration(lines):
    for line_idx, line in enumerate(lines):
        if "BOSS is done!" in line:
            return lines[line_idx-1:line_idx+5]
    return None


def get_timings(lines):
    timings, line_indices = [], []
    for line_idx, line in enumerate(lines):
        if "Total time" in line:
            timings.append(float(line.split()[-1]))
            line_indices.append(line_idx)
    return timings, line_indices


def calculate_correct_timings(timings):
    timings = np.array(timings)
    for timing_idx in range(1, len(timings)):
        if timings[timing_idx] < timings[timing_idx-1]:
            time_delta = timings[timing_idx-1]
            timings[timing_idx:] += time_delta
    return timings


def correct_timings(lines, timings):
    timings_idx = 0
    corrected_lines = []
    for line_idx, line in enumerate(lines):
        if "Total time" in line:
            line = line.split()
            line[-1] = str(round(timings[timings_idx], 3))
            line[3] += '    '
            line = ' '.join(line)
            timings_idx += 1
        corrected_lines.append(line + '\n')
    return corrected_lines


if __name__ == '__main__':
    main()
