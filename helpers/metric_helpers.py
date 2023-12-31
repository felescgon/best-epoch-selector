import copy
import os
import re
import json
import csv
from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from epoch_selection.best_epochs_selector import BestEpochsSelector
from epoch_selection.similarity_copy import SimilarityCopy
from helpers.reader_utils import get_epoch_parent_path, get_epochs_from_experiment, load_ts_from_path
from helpers.similarity_ts_utils import create_similarity_ts_config


def compute_metrics(arguments, header_ts1, ts1, experiment_directories, save_directory_folder):
    print('Computing metrics...')
    if arguments.window_selection_metric not in arguments.metrics:
        arguments = copy.deepcopy(arguments)
        arguments.metrics.append(arguments.window_selection_metric)
    for experiment_directory in tqdm(experiment_directories, total=len(experiment_directories), colour='green'):
        metric_results_by_epoch = {}
        epoch_directories = get_epochs_from_experiment(experiment_directory)
        sorted_epoch_directories = natsorted(epoch_directories)
        tqdm_epoch_iterator = tqdm(sorted_epoch_directories, total=len(sorted_epoch_directories), desc=__get_experiment_name(experiment_directory), leave=False)
        for epoch_directory in tqdm_epoch_iterator:
            metric_results_by_epoch.update(__get_metrics_results_by_epoch(arguments, header_ts1, ts1, epoch_directory, tqdm_epoch_iterator))
            __save_experiment_to_csv(save_directory_folder, __get_experiment_name(experiment_directory), 
                                     __get_epoch_name(epoch_directory), metric_results_by_epoch[get_epoch_parent_path(epoch_directory)])
        experiment_selector = BestEpochsSelector(metric_results_by_epoch, 'experiment_dir_name')
        best_experiments = experiment_selector.select_best_epochs(arguments.metrics_to_compare, arguments.n_best)
        __save_selected_experiments(save_directory_folder, best_experiments, arguments.n_best)


def __get_experiment_name(experiment_directory):
    if 'model' in experiment_directory:
        return experiment_directory.split(os.path.sep)[-2]
    return experiment_directory.split(os.path.sep)[-1]


def __get_epoch_name(epoch_directory):
    return epoch_directory.split(os.path.sep)[-2].split()[0]


def __get_metrics_results_by_epoch(arguments, header_ts1, ts1, epoch_directory, tqdm_epoch_iterator):
    metric_results_by_epoch = {}
    results_path = f'{get_epoch_parent_path(epoch_directory)}/results.json'
    if not arguments.recompute_metrics and os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as results:
                metrics_results = json.load(results)
                if set(arguments.metrics).issubset(set(metrics_results['Aggregated'].keys())):
                    metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = metrics_results
                    __update_metrics_path(metrics_results, epoch_directory)
                    tqdm_epoch_iterator.set_postfix(epoch=__get_epoch_name(epoch_directory), status='Skipping...')
                else:
                    __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory)
                    tqdm_epoch_iterator.set_postfix(epoch=__get_epoch_name(epoch_directory), status='Computing...')
        except json.decoder.JSONDecodeError:
            __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory)
            tqdm_epoch_iterator.set_postfix(epoch=__get_epoch_name(epoch_directory), status='Computing...')
    else:
        __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory)
        tqdm_epoch_iterator.set_postfix(epoch=__get_epoch_name(epoch_directory), status='Computing...')

    return metric_results_by_epoch


def __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory):
    ts2_dict = load_ts_from_path(epoch_directory)
    if len(ts2_dict.values()) != 0:
        similarity_ts_config = create_similarity_ts_config(arguments, list(ts2_dict.keys()), header_ts1)
        similarity_ts = SimilarityCopy(ts1, list(ts2_dict.values()), similarity_ts_config)
        if similarity_ts_config.metric_config.metrics:
            metric_results_by_epoch.update(__get_metrics_results_by_samples(epoch_directory, similarity_ts))
    else:
        print(f'No time series found in {epoch_directory}.')


def __get_metrics_results_by_samples(epoch_directory, similarity_ts):
    metric_results_by_samples = {}
    metric_results_by_samples[get_epoch_parent_path(epoch_directory)] = {}
    metric_results_by_samples[get_epoch_parent_path(epoch_directory)] = __compute_metrics_by_samples(similarity_ts, epoch_directory)
    return metric_results_by_samples


def __compute_metrics_by_samples(similarity_ts, save_directory_path):
    metrics_results_by_samples = {"Aggregated": {}, "Individual": {}}
    metric_computer_iterator = similarity_ts.get_metric_computer()
    metrics_sums = defaultdict(float)
    metrics_counts = defaultdict(int)
    for filename, metric_name, computed_metric in metric_computer_iterator:
        if metrics_results_by_samples["Individual"].get(filename) is None:
            metrics_results_by_samples["Individual"][filename] = {}
        metrics_results_by_samples["Individual"][filename][metric_name] = {}
        metrics_results_by_samples["Individual"][filename][metric_name] = computed_metric
        metrics_sums[metric_name] += computed_metric['Multivariate']
        metrics_counts[metric_name] += 1
    for metric_name in metrics_sums:
        metrics_results_by_samples["Aggregated"][metric_name] = metrics_sums[metric_name] / metrics_counts[metric_name]
    __save_metrics(json.dumps(metrics_results_by_samples, indent=4, ensure_ascii=False).encode('utf-8'),
                   path=f'{os.path.dirname(save_directory_path)}')
    return metrics_results_by_samples


def __save_metrics(computed_metrics, path):
    try:
        os.makedirs(f'{path}', exist_ok=True)
        with open(f'{path}/results.json', 'w', encoding='utf-8') as file:
            file.write(computed_metrics.decode('utf-8'))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the metrics in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __update_metrics_path(computed_metrics, path):
    path_components = os.path.normpath(path).split(os.sep)
    path = os.sep.join(path_components).replace(os.path.sep, '/')
    results_save_path = '/'.join(path.split('/')[:-1])
    new_individual_metrics = {}
    for sample_path, sample_metrics in computed_metrics['Individual'].items():
        sample_name = sample_path.split('/')[-1]
        new_sample_path = f'{path}/{sample_name}'
        new_individual_metrics[new_sample_path] = sample_metrics
    computed_metrics.pop('Individual')
    computed_metrics['Individual'] = new_individual_metrics
    try:
        with open(f'{results_save_path}/results.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(computed_metrics, indent=4, ensure_ascii=False))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the metrics in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __save_experiment_to_csv(save_directory_path, experiment_name, epoch_name, epoch_data_dict):
    try:
        save_file_path_csv = f'{save_directory_path}/every_experiment.csv'
        does_file_exist = os.path.exists(save_file_path_csv)
        if not os.path.exists(f'{save_directory_path}'):
            os.makedirs(f'{save_directory_path}', exist_ok=True)
        with open(save_file_path_csv, 'a', encoding='utf-8') as file:
            writer = csv.writer(file)
            row_data =  parse_file_name_to_dict(experiment_name)
            row_data.update({'epoch': int(epoch_name.split('_')[-1]), **epoch_data_dict['Aggregated']})
            if not does_file_exist:
                header = row_data.keys()
                writer.writerow(header)
            writer.writerow(row_data.values())
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the best experiments in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def parse_file_name_to_dict(input_string):
    components = re.findall(r'([a-zA-Z_]+)[-_](\d+(?:\.\d+)?)', input_string)
    data_dict = {}
    for key, value in components:
        data_dict[key] = float(value) if '.' in value else int(value)
    return data_dict


def __save_selected_experiments(save_directory_path, experiments, n_best):
    try:
        save_file_path_json = f'{save_directory_path}/{n_best}_epochs_by_experiment.json'
        save_file_path_csv = f'{save_directory_path}/{n_best}_epochs_by_experiment.csv'
        experiments_to_save = json.dumps(experiments, indent=4, ensure_ascii=False).encode('utf-8')
        if os.path.exists(save_file_path_json):
            experiments_to_save = __load_sorted_experiments_from_file(experiments, save_file_path_json)
        with open(save_file_path_json, 'w', encoding='utf-8') as file:
            file.write(experiments_to_save.decode('utf-8'))
        __write_selected_experiments_to_csv(save_file_path_csv, save_file_path_json)
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the best experiments in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __load_sorted_experiments_from_file(experiments, save_file_path):
    with open(save_file_path, 'r', encoding='utf-8') as file:
        already_saved_experiments = json.load(file)
        already_saved_experiments.update(experiments)
        sorted_experiments = {
                    epoch: already_saved_experiments[epoch] for epoch in natsorted(already_saved_experiments,
                        key=lambda epoch: already_saved_experiments[epoch]['intersection_epochs_n'])
                }
        experiments_to_save = json.dumps(sorted_experiments, indent=4, ensure_ascii=False).encode('utf-8')
    return experiments_to_save


def __write_selected_experiments_to_csv(file_path, json_file_path):
    with open(json_file_path, 'r', newline='', encoding='utf-8') as json_file:
        experiments = json.load(json_file)
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        field_names = ['experiment'] + ['epoch'] + __get_metrics(experiments) + [key for key in list(experiments.values())[0] if key not in ['best_epochs']]
        writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=';')
        writer.writeheader()
        for experiment_name, experiment_data in experiments.items():
            row_data = {}
            row_data['experiment'] = experiment_name
            row_data.update({column: experiment_data[column] for column in experiment_data.keys() if not isinstance(experiment_data[column], dict)})
            for best_epoch, metric_values in experiment_data['best_epochs'].items():
                __add_metric_columns(best_epoch, metric_values, row_data)
                writer.writerow(row_data)


def __add_metric_columns(best_epoch, metric_values, row_data):
    row_data['epoch'] = best_epoch.split('_')[-1]
    for metric_name, metric_value in metric_values.items():
        row_data[metric_name] = metric_value


def __get_metrics(d):
    fourth_level_keys = set()
    for value in d.values():
        if isinstance(value, dict):
            for inner_value in value.values():
                if isinstance(inner_value, dict):
                    for deeper_inner_value in inner_value.values():
                        if isinstance(deeper_inner_value, dict):
                            fourth_level_keys.update(deeper_inner_value.keys())
    return list(fourth_level_keys)
