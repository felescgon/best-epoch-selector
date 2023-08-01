import os
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
    for experiment_directory in experiment_directories:
        metric_results_by_epoch = {}
        epoch_directories = get_epochs_from_experiment(experiment_directory)
        tqdm_epoch_iterator = tqdm(epoch_directories, total=len(epoch_directories), desc=experiment_directory)
        for epoch_directory in tqdm_epoch_iterator:
            tqdm_epoch_iterator.set_postfix(epoch=epoch_directory.split(os.path.sep)[-2].split()[0], status='Computing...')
            metric_results_by_epoch.update(__get_metrics_results_by_epoch(arguments, header_ts1, ts1, epoch_directory, tqdm_epoch_iterator))
        experiment_selector = BestEpochsSelector(metric_results_by_epoch, 'experiment_dir_name')
        best_experiments = experiment_selector.select_best_epochs(arguments.metrics_to_compare, arguments.n_best)
        __save_selected_experiments(save_directory_folder, best_experiments, arguments.n_best)


def __get_metrics_results_by_epoch(arguments, header_ts1, ts1, epoch_directory, tqdm_epoch_iterator):
    metric_results_by_epoch = {}
    if not arguments.recompute_metrics and os.path.exists(f'{get_epoch_parent_path(epoch_directory)}/results.json'):
        with open(f'{get_epoch_parent_path(epoch_directory)}/results.json', 'r', encoding='utf-8') as results:
            metrics_results = json.load(results)
            if set(arguments.metrics).issubset(set(metrics_results['Aggregated'].keys())):
                metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = metrics_results
                tqdm_epoch_iterator.set_postfix(epoch=epoch_directory.split(os.path.sep)[-2].split()[0], status='Skipping...')
            else:
                __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory)
    else:
        __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory)
    return metric_results_by_epoch


def __compute_metrics_by_epoch(arguments, header_ts1, ts1, metric_results_by_epoch, epoch_directory):
    ts2_dict = load_ts_from_path(epoch_directory)
    if len(ts2_dict.values()) != 0:
        similarity_ts_config = create_similarity_ts_config(arguments, list(ts2_dict.keys()), header_ts1)
        similarity_ts = SimilarityCopy(ts1, list(ts2_dict.values()), similarity_ts_config)
        if similarity_ts_config.metric_config.metrics:
            metric_results_by_epoch.update(__get_metrics_results_by_samples(epoch_directory, similarity_ts, arguments.recompute_metrics))
    else:
        print(f'No time series found in {epoch_directory}.')


def __get_metrics_results_by_samples(epoch_directory, similarity_ts, recompute_metrics):
    metric_results_by_samples = {}
    metric_results_by_samples[get_epoch_parent_path(epoch_directory)] = {}
    if not recompute_metrics and os.path.exists(f'{get_epoch_parent_path(epoch_directory)}/results.json'):
        with open(f'{get_epoch_parent_path(epoch_directory)}/results.json', 'r', encoding='utf-8') as results:
            metrics_results = json.load(results)
            if set(similarity_ts.similarity_ts_config.metric_config.metrics).issubset(set(metrics_results['Aggregated'].keys())):
                metric_results_by_samples[get_epoch_parent_path(epoch_directory)] = metrics_results
                print(f'\nEpoch {os.path.basename(get_epoch_parent_path(epoch_directory))} already computed. Skipping...')
            else:
                metric_results_by_samples[get_epoch_parent_path(epoch_directory)] = __compute_metrics_by_samples(similarity_ts, epoch_directory)
    else:
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
                   path=f'{save_directory_path}/../')
    return metrics_results_by_samples


def __save_metrics(computed_metrics, path='results/metrics'):
    try:
        os.makedirs(f'{path}', exist_ok=True)
        with open(f'{path}/results.json', 'w', encoding='utf-8') as file:
            file.write(computed_metrics.decode('utf-8'))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the metrics in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __save_selected_experiments(save_directory_path, experiments, n_best):
    try:
        save_file_path_json = f'{save_directory_path}/{n_best}_epochs_by_experiment.json'
        save_file_path_csv = f'{save_directory_path}/{n_best}_epochs_by_experiment.csv'
        experiments_to_save = json.dumps(experiments, indent=4, ensure_ascii=False).encode('utf-8')
        if os.path.exists(f'{save_directory_path}'):
            experiments_to_save = __load_sorted_experiments_from_file(experiments, save_file_path_json)
        else:
            os.makedirs(f'{save_directory_path}', exist_ok=True)
        with open(save_file_path_json, 'w', encoding='utf-8') as file:
            file.write(experiments_to_save.decode('utf-8'))
        __write_experiments_to_csv(save_file_path_csv, save_file_path_json)
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


def __write_experiments_to_csv(file_path, json_file_path):
    with open(json_file_path, 'r', newline='', encoding='utf-8') as json_file:
        experiments = json.load(json_file)
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        field_names = ['experiment'] + list(set(key for nested_dict in experiments.values() for key in nested_dict.keys()))
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for key, nested_dict in experiments.items():
            row_data = {'experiment': key, **nested_dict}
            writer.writerow(row_data)
