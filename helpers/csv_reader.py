import json
import os
import csv
import numpy as np
import pandas as pd
from natsort import natsorted

def load_df_from_csv(filename):
    return pd.read_csv(filename, delimiter=__detect_line_delimiter(filename), dtype=object)


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


def __check_headers(header_ts1, header_ts2):
    if header_ts1 != header_ts2:
        raise ValueError('All time series must have the same header column names.')


def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = __detect_line_delimiter(filename)
    header = __read_header_from_csv(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0
    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header


def load_ts_from_path(path, header_ts1, has_header=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f' Path {path} does not exist.')
    time_series = {}
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
                        ts2, header_ts2 = load_ts_from_csv(file_path, has_header)
                        __check_headers(header_ts1, header_ts2)
                        time_series[os.path.join(root, file).replace(os.path.sep, '/')] = ts2
    return time_series


def json_file_to_experiments_df(file):
    with open(f'{file}/metrics/results.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    flattened_data = {key: flatten_json(value) for key, value in json_data.items()}
    experiments_df = pd.DataFrame(flattened_data).T.reset_index()
    experiments_df = experiments_df.rename(columns={'index': 'experiment_dir_name'})
    experiments_df.columns = experiments_df.columns.str.replace('_Multivariate', '')
    experiments_df['experiment_dir_name'] = experiments_df['experiment_dir_name'].apply(os.path.dirname)
    return experiments_df

def flatten_json(json_data):
    flat_data = {}
    for key, value in json_data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat_data[key + "_" + k] = v
        else:
            flat_data[key] = value
    return flat_data
