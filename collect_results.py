"""
Open results and report best hyperparameters and results

Example usage:
python collect_results.py --dirs MAS_I_seeds MAS_II_seeds
"""

import os
import copy
import json
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt


def get_dir_list(dirs):
    all_directories = []
    for exper_dir in dirs:
        # List all subfolders of each of the directory list (dirs)
        all_folders = [
            dI for dI in os.listdir(exper_dir)
            if os.path.isdir(os.path.join(exper_dir, dI))
        ]
        all_directories += [os.path.join(exper_dir, folder)
                            for folder in all_folders]
    return all_directories


def collect_results(dirs, data_files):
    # First construct a file list of the given experiment
    all_directories = get_dir_list(dirs)

    # Open all results and get summaries
    print('Processing data...')
    all_data_lists = [{} for _ in data_files]

    for run_folder in all_directories:
        for idx, file_ in enumerate(data_files):
            file_path = os.path.join(run_folder, file_)
            with open(file_path, 'r') as f:
                data = json.load(f)
            for metric_name in data.keys():
                if metric_name not in all_data_lists[idx].keys():
                    all_data_lists[idx][metric_name] = []
                all_data_lists[idx][metric_name] += [data[metric_name]]

    return all_data_lists


def best_hyperparameters(dirs, selection_criterion=None, forced_metric=None,
                         verbose=False, do_plot=False, silent=False):
    exper_name = ''
    # Extract the configuration dictionaries
    all_directories = get_dir_list(dirs)
    selected_dirs = all_directories
    config_dicts = []
    for run_dir in all_directories:
        config_file = os.path.join(run_dir, 'config.txt')
        if os.path.exists(config_file):
            with open(config_file) as json_file:
                config_dict = json.load(json_file)
                config_dict['save_dir'] = run_dir
                config_dicts.append(config_dict)

    class MyError(Exception):
        pass

    if selected_dirs == []:
        raise MyError("No runs found for experiment {}".format(exper_name))

    """
    Then we identify altered hyperparameters in cofig_dicts,
    and log their values: they can be columns of the table
    """
    ignore_columns = [
        'save_dir', 'grad_accum_steps', 'dev_evaluate_model',
        'evaluate_model', 'wg_evaluate', 'comve_evaluate', 'use_devices']
    column_dict_values = {}
    for key in config_dicts[0].keys():
        if key not in ignore_columns:
            values_set = set([
                config_dict[key] for config_dict in config_dicts
            ])
            if len(values_set) > 1:
                column_dict_values[key] = list(values_set)
                column_dict_values[key].sort()
    if column_dict_values == {}:
        raise MyError("No columns found for experiment {} ".format(exper_name))

    # Extract the results of the experiment:
    all_result_dicts = []
    for run_dir in selected_dirs:
        if selection_criterion == 'snli_ppl':
            results_file = 'final_esnli_dev_results.json'
        elif selection_criterion == 'wg_ppl':
            results_file = 'final_wg_nles_dev_results.json'
        elif selection_criterion == 'wg_acc':
            results_file = 'final_wg_acc_dev_results.json'
        elif selection_criterion == 'comve_ppl':
            results_file = 'final_comve_nles_dev_results.json'
        elif selection_criterion == 'comve_acc':
            results_file = 'final_comve_acc_dev_results.json'
        else:
            raise NotImplementedError
        results_path = os.path.join(run_dir, results_file)

        if os.path.exists(results_path):
            with open(results_path) as json_file:
                all_result_dicts.append(json.load(json_file))
        else:
            all_result_dicts.append({})

    # Get the list of available metrics
    metrics = None
    for results_dict in all_result_dicts:
        if len(results_dict.keys()) > 0:
            metrics = list(results_dict.keys())
            break
    if metrics is None:
        raise MyError('')
    if forced_metric is not None:
        metric = forced_metric
    elif len(metrics) > 1:
        print('Available metrics for reporting:', metrics)
        print('Please type one of the given metrics...')
        metric = input()
    else:
        metric = metrics

    columns = list(column_dict_values.keys())
    if not silent:
        pp = pprint.PrettyPrinter(indent=2)
        print('Hyperparameter space:')
        pp.pprint({k: column_dict_values[k] for k in columns})
    column_sizes = [len(column_dict_values[col]) for col in columns]
    # Values of the metric for all hyperparameter combinations:
    values_array = np.zeros(column_sizes, dtype=float)

    # Iterate over the array and fill-in the results
    it = np.nditer(values_array, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Construct indices for each hyperparameter:
        col_value_dict = {}
        for idx, value in enumerate(it.multi_index):
            column = columns[idx]
            col_value_dict[column] = column_dict_values[column][value]
        # Retrieve the result from hyperparameter combination
        # corresponding to it.multi_index.
        value = None
        for config_dict, results_dict in zip(config_dicts, all_result_dicts):
            if all(config_dict[key] == col_value_dict[key]
                   for key in col_value_dict.keys()):
                if metric in results_dict.keys():
                    value = results_dict[metric]
                    if value > 1e10:
                        value = 1.0e10 - 1
                else:
                    value = None
        values_array[it.multi_index] = value
        it.iternext()

    if np.isnan(np.sum(values_array)):
        print('\n   WARNING: nan value found. See verbose mode.\n')

    if verbose:
        print('Results:')
        pp = pprint.PrettyPrinter(indent=2, width=80)
        pp.pprint(values_array)

    if 'ewg_train_data_path' in columns and (
            'acc' in metric or 'Acc' in metric):
        print('Forcing new_train.jsonl results only (column id = 1)')
        print(column_dict_values['ewg_train_data_path'])
        values_array = (
            values_array[0] if column_dict_values['ewg_train_data_path'][0]
            == 'new_train.jsonl' else values_array[1])
        columns = [col for col in columns if col != 'ewg_train_data_path']

    # Now take argmax to find the best hyperparam-s
    if 'perplexity' in metric or 'ppl' in metric:
        selected_ids = np.unravel_index(np.nanargmin(values_array),
                                        values_array.shape)
    else:
        selected_ids = np.unravel_index(np.nanargmax(values_array),
                                        values_array.shape)

    if not silent:
        print('Best hyperparameters:')
    selected_column_dict_values = {}
    best_hyps = {}
    for value_idx, column in zip(selected_ids, columns):
        value = column_dict_values[column][value_idx]
        selected_column_dict_values[column] = value
        best_hyps[column] = value
        if not silent:
            print(column, value)

    best_value = values_array[selected_ids]
    if not silent:
        print('Best {}:'.format(metric), best_value)

    # Retrieve the best run's folder:
    # Select those runs that have the correct column_dict_values:
    continue_signal = False
    best_run_folder = ''
    for config_dict in config_dicts:
        for key in selected_column_dict_values.keys():
            if config_dict[key] != selected_column_dict_values[key]:
                continue_signal = True
        if not continue_signal:
            best_run_folder = copy.deepcopy(config_dict['save_dir'])
        continue_signal = False
    if not silent:
        print('Best run:', best_run_folder)

    if do_plot:
        # Plot all metrics one by one
        all_axes = range(values_array.ndim)
        for column_id in all_axes:
            print('Plotting', columns[column_id])
            # min/max over the rest of the hyperparameters (note: axes=columns)
            other_axes = tuple(set(all_axes)- {column_id})
            if 'perplexity' in metric or 'ppl' in metric:
                y_points = np.nanmin(values_array, axis=other_axes)
            else:
                y_points = np.nanmax(values_array, axis=other_axes)
            plt.clf()  # Clear the plot
            plt.plot(range(len(y_points)), y_points, linestyle='dotted')
            # Rename x-axis scale by the hyperparameter values
            plt.xticks(range(len(y_points)),
                       column_dict_values[columns[column_id]])
            plt.xlabel(columns[column_id])
            plt.ylabel(metric)
            plt.show()
            plt.savefig('{}.png'.format(columns[column_id]))

    return {
        'selection_criterion': selection_criterion,
        'best_value': best_value,
        'best_hyperparams': best_hyps
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dirs", default=None, nargs='+', required=True,
        help="The experiment directories to summarize, "
             "with runs in sub-folders")
    parser.add_argument(
        "--selection_criterion", choices=[
            'snli_ppl', 'wg_ppl', 'wg_acc', 'comve_ppl', 'comve_acc'],
        required=True, help="The hyperparameter selection criterion")
    parser.add_argument(
        "--verbose", '-v', action='store_true',
        help="print stuff")
    parser.add_argument(
        "--do_plot", '-plot', action='store_true',
        help="plot the loss landscape")
    args = parser.parse_args()

    best_hyperparameters(
        dirs=args.dirs, verbose=args.verbose, do_plot=args.do_plot,
        selection_criterion=args.selection_criterion)
