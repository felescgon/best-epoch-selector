import os
import csv
import pandas as pd

def load_df_from_csv(filename):
    return pd.read_csv(filename, delimiter=__detect_line_delimiter(filename))


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
