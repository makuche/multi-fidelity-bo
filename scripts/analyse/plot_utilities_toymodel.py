import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', **{ 'family': 'serif', 'size': 12, })
plt.rc('text', **{ 'usetex': True, 'latex.preamble': r""" \usepackage{physics} \usepackage{siunitx} """ })
import seaborn as sns
import click

from collections import defaultdict, OrderedDict
from pathlib import Path

from src.read_toymodel_outputs import OutputFileParser, ParserToDataFrame

THESIS_FOLDER = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = THESIS_FOLDER / 'results/figs'
TOYMODEL_FOLDER = THESIS_FOLDER / 'data/multi_task_learning/toymodel'
legend_labels = ['BL'] + [f'MFBO {i}' for i in range(1,8)]
transfer_learning_results = {'uhf_hf': 6, 'uhf_lf': 17}
scales = {'uhf_hf': [2, 40], 'uhf_lf': [2, 55]}
fidelities = 'uhf_hf'
plot_settings = {
    'fidelities':fidelities,
    'dimension':2,
    'tolerance':0.23,
    'convergence_metric':'convergence_cost',
    'runs':10,
    'single_task_cost':12000,
    'scale_y_axis':True,
    'strategy_indices':[*range(1,7), 10],
    'plot_mumbo':True,
    'print_nonconverged_runs':True,
    'artificial_cost':[12000, 30],
    'legend_labels':legend_labels,
    'plot_bounds': scales[fidelities],
    'best_tl_result':None,   # 4 HF->UHF, 15 LF->UHF
    'acqfns': ['elcb', 'mes'],
    'acqfns_label': {'elcb': 'ELCB', 'mes': 'MES', 'mumbo': 'MUMBO'}
}
titles = {'uhf_lf': r'LF $\rightarrow$ UHF',
          'uhf_hf': r'HF $\rightarrow$ UHF'}
@click.command()
@click.option('--show_plots', default=False, is_flag=True,
              help='Show (and don\'t save) plots.')
@click.option('--fidelities', default='uhf_hf', type=str,
              help="Chose between 'uhf_hf' or 'uhf_lf'.")
def main(show_plots, fidelities):
    plot_settings['fidelities'] = fidelities

    plot_cost_to_reach_convergence(plot_settings, TOYMODEL_FOLDER, show_plots)

def get_discretized_dict(dictionary, rounding_function):
    """
    This utility function returns a dictionary, where the keys
    of the original dictionary are 'discretized' and the values
    of the previous keys are collected and paired with the
    discretized keys.

    Examples for the rounding_function:
        1) round_to_half = lambda x: round(x * 2) / 2, or
        2) round_to_int = lambda x: round(x)
        3) def round_to_multiple(number, multiple=12000/4):
               return int(np.ceil(number/multiple)*multiple)
    """
    ordered_dict, rounded_dict = OrderedDict(), defaultdict(list)
    for key in sorted(dictionary.keys()):
        ordered_dict[key] = dictionary[key]
    rounded_keys = [rounding_function(key) for key in dictionary.keys()]
    cost_to_rounded_cost_map = {
        key: val for key, val in zip(dictionary.keys(), rounded_keys)}
    for key in dictionary.keys():
        rounded_key = cost_to_rounded_cost_map[key]
        rounded_dict[rounded_key] += dictionary[key]
    rounded_sorted_dict = OrderedDict()
    for key in sorted(rounded_dict.keys()):
        rounded_sorted_dict[key] = rounded_dict[key]
    return rounded_sorted_dict


def round_to_multiple(number, multiple=12000/4):
    return int(np.ceil(number/multiple)*multiple)


def plot_singletask_sample_locations(experiment, num_experiments, folder):
    """Plot single-task sample locations in search space.

    Parameters
    ----------
    experiment : str
        e.g. 'uhf_2d_mes_st'
    num_experiments : int
        Amount of plots to be generated, e.g. 1.
    folder : str
        e.g. 'out', path of the output files.
    """
    fig = plt.figure()
    for exp_idx in range(num_experiments):
        parser = OutputFileParser(f'{experiment}_run{exp_idx}', folder)
        alphas = np.linspace(0.1, 1, len(parser.data['xy']))[::-1]
        plt.scatter(
            parser.data['xy'][:, 0], parser.data['xy'][:, 1], alpha=alphas)
        plt.title(f'Setup: {experiment}', fontsize=16)
        plt.show()


def plot_multitask_sample_locations_with_bincounts(
        experiment, folder, samples_over_iteration=False):
    """Plot single-task sample locations in search space.

    Parameters
    ----------
    experiment : str
        e.g. 'uhf_lf_2d_elcb_strategy1_run0'
    folder : str
        e.g. 'out', path of the output files.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    parser = OutputFileParser(f'{experiment}', folder)
    samples, sample_indices = parser.data['xy'], parser.data['sample_indices']
    dimension = samples.shape[1] - 1
    gmp = parser.data['gmp']
    predicted_minimum_x = gmp[:, 0:2]
    colors = ['blue' if idx_ == 0 else 'red' for idx_ in sample_indices]
    alphas = np.linspace(0.1, 1.0, len(colors))[::-1]
    axs[0].scatter(samples[:, 0], samples[:, 1], c=colors, alpha=alphas)
    axs[0].set_title('Sample locations (higher intensity: earlier sample locations).\nBlue: High fidelity, Red: low fidelity',
                     fontsize=16)
    color_appearances = np.bincount(sample_indices)
    total_num_samples = len(sample_indices)

    # Plot bincounts after 20, 40, 60, 80, 100% of number of samples
    bin_count_ranges = [int(quantile*total_num_samples)
                        for quantile in [0.2, 0.4, 0.6, 0.8, 1.0]]
    bin_counts = np.array([
        np.bincount(sample_indices[:num_samples])
        for num_samples in bin_count_ranges])
    axs[1].bar(range(len(bin_counts)), bin_counts[:, 0],
                color='blue', label='HF', bottom=bin_counts[:, 1])
    axs[1].bar(range(len(bin_counts)), bin_counts[:, 1], color='red',
                label='LF')

    # Plot vertical convergence line
    df = ParserToDataFrame(parser)()
    convergence_idx = df['convergence_idx'][0]
    if convergence_idx is not None:
        num_samples = len(df['sample_indices'][0])
        axs[1].axvline(convergence_idx/num_samples * len(bin_counts) - 1,
                       color='gray', linestyle='--', zorder=5,
                       label='Convergence\nreached')

    axs[1].set_xticks(range(5))
    axs[1].set_xticklabels(['20%', '40%', '60%', '80%', '100%'])
    axs[1].set_ylabel('Number of samples after\n % total iterations',
                        fontsize=14)
    axs[1].set_xlabel('% of number of total iterations')
    axs[1].set_title('Proportion of LF&HF samples', fontsize=16)
    axs[1].legend(fontsize=14)
    fig.suptitle(f'Setup: {experiment}', fontsize=22, y=1.05)
    plt.show()

    if samples_over_iteration:
        fig = plt.figure(figsize=(16, 7))
        # List of 10 colors for plotting
        colors = ['green', 'orange', 'purple', 'brown', 'red', 'blue']
        for dim_idx in range(dimension):
            plt.plot(
                predicted_minimum_x[:, dim_idx],
                label=f'Predicted minimum Dimension {dim_idx}',
                color=colors[dim_idx], lw=3)
            plt.scatter(range(total_num_samples), samples[:, dim_idx],
                        color=colors[dim_idx], s=50, alpha=.7,
                        label=f'Sample at Dimension {dim_idx}')
        for iteration_idx, sample_idx in enumerate(sample_indices):
            label = 'HF Sample' if sample_idx == 0 else 'LF Sample'
            alpha = .15 if sample_idx == 1 else .4
            plt.axvspan(iteration_idx-.5, iteration_idx+.5,
                        facecolor='gray', alpha=alpha, label=label)
        plt.xlim(0, total_num_samples)
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_by_label = dict(zip(labels, handles))
        plt.legend(legend_by_label.values(), legend_by_label.keys(),
                   bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize=18)
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel(r'$x_i$ Sample location', fontsize=18)
        plt.show()

def plot_timing_data(experiments, folder, labels, artificial_acq_times=False):
    fig = plt.figure(figsize=(9, 6))
    parsers = [OutputFileParser(f'{experiment}', folder) for experiment
               in experiments]
    for parser, label in zip(parsers, labels):
        if artificial_acq_times:
            costs = parser.data['acqcost']
            indices = parser.data['sample_indices']
            parser.data['acq_times'] = np.array([costs[0] if idx == 0 else costs[1]
                                        for idx in indices])
        for array in [parser.data['acq_times'], parser.data['iter_times']]:
            array = np.array(array)
        parser.data['iter_times'] = np.array(parser.data['iter_times'])
        parser.data['total_times'] = np.cumsum(parser.data['acq_times'] + \
            parser.data['iter_times'])
        split = label.split('_')
        label_name = split[0] + ' with support ' + split[1]
        plt.plot(parser.data['iter_times'], label=label_name)

        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('Iteration time (s)', fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

def plot_strategies(fidelities, acqfn, folder='out', range_bound=10,
                    plot_filling=True):
    fig = plt.figure(figsize=(16, 9))
    cost_regret_pairs = defaultdict(list)

    # Single-task
    for idx in range(10):
        parser = OutputFileParser(f'uhf_2d_{acqfn}_st_run{idx}', folder=folder)
        ypred = np.array(parser.data['gmp'])[:, -2]
        initpts = parser.data['initpts']
        cumulative_costs = parser.data['cumulative_cost'][initpts-1:]
        for cost, regret in zip(cumulative_costs, ypred):
            cost_regret_pairs[float(round(cost, 1))].append(regret)
    for cost in cost_regret_pairs:
        cost_regret_pairs[cost] = np.mean(cost_regret_pairs[cost]), \
            np.sqrt(np.var(cost_regret_pairs[cost]))
    costs = np.array(list(cost_regret_pairs.keys()))
    #true_min = np.array(list(cost_regret_pairs.values()))[-1, 0]
    true_min = -202861.3237
    regrets = np.array(list(cost_regret_pairs.values()))
    regret_means, regrets_sd = regrets[:, 0].squeeze(), regrets[:, 1].squeeze()
    plt.plot(costs, regret_means-true_min, c='gray')
    if plot_filling:
        plt.fill_between(costs, regret_means-true_min - 1.96*regrets_sd,
                         regret_means-true_min + 1.96*regrets_sd,
                         color='gray', alpha=.2)

    strategies = range(1, 7)
    for strategy, color in zip(strategies, ['b', 'g', 'r', 'c', 'm', 'y']):
        unordered_cost_regret_pairs = defaultdict(list)
        for idx in range(range_bound):
            parser = OutputFileParser(
                f'{fidelities}_2d_{acqfn}_strategy{strategy}_run{idx}',
                folder=folder)
            ypred = np.array(parser.data['gmp'])[:, -2]
            initpts = parser.data['initpts']
            cumulative_costs = parser.data['cumulative_cost'][initpts-1:]
            for cost, regret in zip(cumulative_costs, ypred):
                unordered_cost_regret_pairs[float(round(cost, 1))].append(regret)
        rounding_func = lambda x: round(x)
        discretized_dict = get_discretized_dict(
            unordered_cost_regret_pairs, rounding_func)
        costs = np.array([key for key in discretized_dict.keys()])
        regrets = np.array([np.mean(discretized_dict[key])
                            for key in discretized_dict.keys()])
        regrets_sd = np.array(
            [np.sqrt(np.var(discretized_dict[key]))
             for key in discretized_dict.keys()])
        plt.plot(costs, regrets - true_min, c=color, ls='solid',
                 label=f'{fidelities}_2d_{acqfn}_strategy{strategy}')
        if plot_filling:
            plt.fill_between(costs, regrets - true_min - 1.96*regrets_sd,
                             regrets - true_min + 1.96*regrets_sd,
                             color=color, alpha=.2)
    plt.axhline(0.1, ls='dashed', c='gray', alpha=.5)
    plt.axhline(-0.1, ls='dashed', c='gray', alpha=.5)
    plt.legend(fontsize=16)
    plt.ylabel('Regret [kcal/mol]', fontsize=16)
    plt.xlabel('Cumulative cost', fontsize=16)
    plt.show()


def plot_mumbo(fidelities, folder='out', true_min=-202861.3237):
    fig = plt.figure(figsize=(16, 9 ))
    cost_regret_pairs = defaultdict(list)

    # Single-task
    for idx in range(10):
        parser = OutputFileParser(f'uhf_2d_elcb_st_run{idx}', folder)
        ypred = np.array(parser.data['gmp'])[:, -2]
        initpts = parser.data['initpts']
        cumulative_costs = parser.data['cumulative_cost'][initpts-1:]
        for cost, regret in zip(cumulative_costs, ypred):
            cost_regret_pairs[float(round(cost, 1))].append(regret)
    for cost in cost_regret_pairs:
        cost_regret_pairs[cost] = np.mean(cost_regret_pairs[cost]), \
            np.sqrt(np.var(cost_regret_pairs[cost]))
    costs = np.array(list(cost_regret_pairs.keys()))
    regrets = np.array(list(cost_regret_pairs.values()))
    regret_means, regrets_sd = regrets[:, 0].squeeze(), regrets[:, 1].squeeze()
    plt.plot(costs, regret_means-true_min, c='gray')
    plt.fill_between(costs, regret_means-true_min - 1.96*regrets_sd,
                     regret_means-true_min + 1.96*regrets_sd,
                     color='gray', alpha=.2)

    color = 'r'
    unordered_cost_regret_pairs = defaultdict(list)
    for idx in range(10):
        parser = OutputFileParser(
            f'{fidelities}_2d_mumbo_inseparable_run{idx}', folder)
        ypred = np.array(parser.data['gmp'])[:, -2]
        initpts = parser.data['initpts']
        cumulative_costs = parser.data['cumulative_cost'][initpts-1:]
        for cost, regret in zip(cumulative_costs, ypred):
            unordered_cost_regret_pairs[float(round(cost, 1))].append(regret)
    rounding_func = lambda x: round(x)
    discretized_dict = get_discretized_dict(unordered_cost_regret_pairs, rounding_func)
    costs = np.array([key for key in discretized_dict.keys()])
    regrets = np.array([np.mean(discretized_dict[key]) for key in discretized_dict.keys()])
    regrets_sd = np.array(
        [np.sqrt(np.var(discretized_dict[key])) for key in discretized_dict.keys()])
    plt.plot(costs, regrets - true_min, c=color, ls='solid',
             label=f'{fidelities}_2d_mumbo_inseparable')
    plt.fill_between(costs, regrets - true_min - 1.96*regrets_sd, regrets - true_min + 1.96*regrets_sd,
                     color=color, alpha=.2)
    plt.axhline(0.1, ls='dashed', c='gray', alpha=.5)
    plt.axhline(-0.1, ls='dashed', c='gray', alpha=.5)
    plt.legend(fontsize=16)
    plt.ylabel('Regret [kcal/mol]', fontsize=16)
    plt.xlabel('Cumulative cost', fontsize=16)
    plt.show()


def plot_cost_to_reach_convergence(plot_settings, folder, show_plots):

    fidelities = plot_settings['fidelities']
    dimension = plot_settings['dimension']
    baseline = fidelities.split('_')[0]
    support_tasks = fidelities.split('_')[1:]
    tolerance = plot_settings['tolerance']
    convergence_metric = plot_settings['convergence_metric']
    runs = plot_settings['runs']
    single_task_cost = plot_settings['single_task_cost']
    scale_y_axis = plot_settings['scale_y_axis']
    strategy_indices = plot_settings['strategy_indices']
    plot_mumbo = plot_settings['plot_mumbo']
    print_nonconverged_runs = plot_settings['print_nonconverged_runs']
    artificial_cost = plot_settings['artificial_cost']
    legend_labels = plot_settings['legend_labels']
    plot_bounds = plot_settings['plot_bounds']
    best_tl_result = plot_settings['best_tl_result']
    acqfns = plot_settings['acqfns']
    acqfns_label = plot_settings['acqfns_label']
    if convergence_metric not in ['convergence_cost', 'convergence_idx']:
        raise Exception('Invalid convergence metric')

    parsers = []
    for acqfn in acqfns:
        if acqfn == 'ei-min-pred':
            parsers += [OutputFileParser(
                f'{baseline}_{dimension}d_ei_st_run{idx}',
                folder, single_task_cost) for idx in range(runs)]
        else:
            parsers += [OutputFileParser(
                f'{baseline}_{dimension}d_{acqfn}_st_run{idx}',
                folder, single_task_cost) for idx in range(runs)]
        for strategy_idx in strategy_indices:
            if strategy_idx != 10:
                parsers += [OutputFileParser(
                    f'{fidelities}_{dimension}d_{acqfn}_strategy{strategy_idx}_run{idx}',
                    folder) for idx in range(runs)]
            else:
                parsers += [OutputFileParser(
                    f'{fidelities}_{dimension}d_{acqfn}_strategy{strategy_idx}_run{idx}',
                    folder, artificial_cost=artificial_cost)
                    for idx in range(runs)]
    if plot_mumbo:
        parsers += [OutputFileParser(
            f'{fidelities}_{dimension}d_mumbo_inseparable_run{idx}', folder)
            for idx in range(runs)]

    df = ParserToDataFrame(parsers, tolerance=tolerance)()
    for acqfn in acqfns_label:
        df['acqfn'] = df['acqfn'].str.replace(acqfn, acqfns_label[acqfn])
    if scale_y_axis:
        df[convergence_metric] = df[convergence_metric].apply(
            lambda x: x/3600)
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    plot_df = df[['acqfn', 'strategy', convergence_metric]]
    single_task_plot_df = plot_df[plot_df['strategy'] == 'st']
    multi_task_plot_df = plot_df[plot_df['strategy'] != 'st'].copy()
    plot_order = ['st'] + [f'strategy{idx}' for idx in strategy_indices]
    ax = sns.boxplot(x='acqfn', y=convergence_metric, hue='strategy',
                     data=single_task_plot_df, hue_order=plot_order, #whis=[0.25, 0.75],
                     palette="tab10", showmeans=True,
                     meanprops={"marker": "x", "markersize": 6,
                                "markeredgecolor": "black",
                                "markeredgewidth": 1})
    ax = sns.boxplot(x='acqfn', y=convergence_metric, hue='strategy', #whis=[0.25, 0.75],
                     data=multi_task_plot_df, hue_order=plot_order,
                     palette="tab10", showmeans=True,
                     meanprops={"marker": "x", "markersize": 6,
                                "markeredgecolor": "black",
                                "markeredgewidth": 1})
    if best_tl_result is not None:
        plt.axhline(best_tl_result, ls='dashed', c='gray', alpha=.5,
                    label='Best Transfer learning strategy')
    handles, labels = plt.gca().get_legend_handles_labels()
    #handles, labels = handles[:10], labels[:10]

    legend_by_label = dict(zip(legend_labels, handles))
    plt.legend(legend_by_label.values(), legend_by_label.keys(), ncol=2,
               fontsize=10)
    ax.set_ylim(0, 150)
    support_tasks = [task.upper() for task in support_tasks]
    if convergence_metric == 'convergence_cost':
        if scale_y_axis:
            plt.ylim(2*3.5, 150)
        else:
            plt.ylim(single_task_cost*plot_bounds[0],
                     single_task_cost*plot_bounds[1])
        ylabel = f'CPU t [h]'
    else:
        plt.ylim(plot_bounds[0], plot_bounds[1])
        ylabel = f'Total iteration to reach convergence [{tolerance} kcal/mol]'
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('')
    if show_plots:
        plt.show()
    else:
        plt.savefig(FIGS_DIR / f'toymodel_{fidelities}.pdf')

    if print_nonconverged_runs:

        plot_df = df[['acqfn', 'strategy', convergence_metric]]
        not_converged_counter = {}
        for acqfn in [*acqfns_label.values()]:
            for strategy in ['st', 'strategy1', 'strategy2', 'strategy3',
                             'strategy4', 'strategy5', 'strategy6',
                             'strategy10']:
                sub_df = plot_df.loc[
                    plot_df.acqfn == acqfn].loc[plot_df.strategy == strategy]
                not_converged_counter[
                    f'{acqfn}_{strategy}'] = sub_df[
                        convergence_metric].isna().sum()
        if plot_mumbo:
            sub_df = plot_df.loc[plot_df.acqfn == 'mumbo']
            not_converged_counter['mumbo'] = sub_df[
                convergence_metric].isna().sum()

        print("Not converged runs")
        print("setup | failed runs / total runs")
        for key in not_converged_counter:
            print(key + '  |  ' + str(not_converged_counter[key]) + ' / '+ str(runs))


if __name__ == '__main__':
    main()
