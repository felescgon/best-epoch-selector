import os
import csv
import json
import numpy as np
from natsort import natsorted


def __detect_line_delimiter(filename):
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        ts_delimiter = csv.Sniffer().sniff(file.readline()).delimiter
    return ts_delimiter


def get_root_paths(experiment_names):
    return set(experiment_names.apply(__get_experiment_root).to_list())


def __get_experiment_root(filename):
    head, _ = os.path.split(filename)
    return head


def get_experiment_tail(filename):
    _, tail = os.path.split(filename)
    return tail


def __read_header_from_csv(filename, ts_delimiter, has_header):
    if has_header:
        header = np.genfromtxt(filename, delimiter=ts_delimiter, names=has_header, max_rows=1, dtype=str).dtype.names
    else:
        first_row = np.loadtxt(filename, delimiter=ts_delimiter, max_rows=1)
        header = ['column-' + str(i) for i in range(len(first_row))]
    return header


def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = __detect_line_delimiter(filename)
    header = __read_header_from_csv(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0
    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header


def load_ts_from_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f' Path {path} does not exist.')
    if os.path.isfile(path):
        raise ValueError('Path must be a directory.')
    if os.path.isdir(path):
        time_series = {}
        for root, _, files in os.walk(path):
            if root.endswith('generated_data'):
                for file in files:
                    files = natsorted(files)
                    if file.endswith('.csv') and not file.startswith('.'):
                        file_path = os.path.join(root, file)
                        ts2, _ = load_ts_from_csv(file_path, False)
                        time_series[os.path.join(root, file).replace(os.path.sep, '/')] = ts2
    return time_series


def get_ts2s_directories(path):
    print('Looking for experiment directories...')
    if not os.path.exists(path):
        raise FileNotFoundError(f' Path {path} does not exist.')
    if os.path.isfile(path):
        raise ValueError('Path must be a directory.')
    if os.path.isdir(path):
        epoch_directories = []
        for root, _, _ in os.walk(path):
            if root.endswith('generated_data'):
                epoch_directories.append(root)
    return natsorted(epoch_directories)


def get_epoch_parent_path(root_path):
    path_components = os.path.normpath(root_path).split(os.sep)
    new_root_path = os.sep.join(path_components[:-1])
    return new_root_path.replace(os.path.sep, '/')


def get_best_epochs_directories(best_experiments):
    directories_to_be_generated = []
    for parent_directory in best_experiments.keys():
        for epoch_directory in best_experiments[parent_directory]['best_epochs']:
            directories_to_be_generated.append(os.path.join(parent_directory, epoch_directory, 'generated_data'))
    return directories_to_be_generated


def get_best_sample_names(epoch_directory, n_best, window_selection_metric):
    with open(f'{epoch_directory}/../results.json', 'r', encoding='utf-8') as results_file:
        epoch_directory_results = json.load(results_file)
    return __get_top_n_files(epoch_directory_results['Individual'], window_selection_metric, n_best)


def __get_top_n_files(dictionary, parameter, n):
    sorted_files = sorted(dictionary.items(), key=lambda x: x[1][parameter]['Multivariate'])
    top_n_files = sorted_files[:n]
    file_names = [file[0] for file in top_n_files]
    return file_names
