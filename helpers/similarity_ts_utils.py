import os
import json
import copy
from tqdm import tqdm
from collections import defaultdict
from datacentertracesdatasets import loadtraces
from similarity_ts.similarity_ts_config import SimilarityTsConfig
from similarity_ts.metrics.metric_config import MetricConfig
from similarity_ts.plots.plot_config import PlotConfig
from similarity_ts.plots.plot_factory import PlotFactory
from experiments_selection.test_class import TestClass

from helpers.reader_utils import get_epoch_parent_path, load_ts_from_path,load_ts_from_csv


def create_similarity_ts_config(arguments, ts2_names, header_names):
    metric_config = None
    plot_config = None if arguments.timestamp_frequency_seconds is None else PlotConfig(arguments.figures,
                                                                                        arguments.timestamp_frequency_seconds)
    if arguments.metrics is not None or arguments.figures is not None:
        metric_config = MetricConfig(arguments.metrics) if arguments.metrics else MetricConfig([])
        plot_config = PlotConfig(arguments.figures,
                                 arguments.timestamp_frequency_seconds) if arguments.figures else PlotConfig([],
                                                                                                             arguments.timestamp_frequency_seconds)
    return SimilarityTsConfig(metric_config, plot_config, arguments.stride, arguments.window_selection_metric,
                              ts2_names,
                              header_names)


def compute_figures(best_epochs_directories, save_directory_path, arguments):
    header_ts1 = tuple(loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300).columns.to_list())
    ts1 = loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300, format='ndarray')
    tqdm_epoch_iterator = tqdm(best_epochs_directories, total=len(best_epochs_directories), desc='Figures')
    for epoch_directory in tqdm_epoch_iterator:
        tqdm_epoch_iterator.set_postfix(file='/'.join(epoch_directory.split(os.path.sep)[1:-1]))
        ts2_dict = load_ts_from_path(epoch_directory)
        arguments_copy = copy.deepcopy(arguments)
        arguments_copy.figures = [plot for plot in arguments.figures if plot in PlotFactory.get_instance().figures_requires_all_samples]
        similarity_ts_config = create_similarity_ts_config(arguments_copy, list(ts2_dict.keys()), header_ts1)
        similarity_ts = TestClass(ts1, list(ts2_dict.values()), similarity_ts_config)
        for ts2_name, plot_name, generated_plots in similarity_ts.get_plot_computer():
            __save_figures(ts2_name, plot_name, generated_plots, save_directory_path)
        top_n_files = get_best_sample_names(epoch_directory, arguments.n_best, arguments.window_selection_metric)
        for filename in top_n_files:
            ts2 = load_ts_from_csv(filename, False)
            arguments_copy = copy.deepcopy(arguments)
            arguments_copy.figures = [plot for plot in arguments.figures if plot not in PlotFactory.get_instance().figures_requires_all_samples]
            similarity_ts_config = create_similarity_ts_config(arguments_copy, [filename], header_ts1)
            similarity_ts = TestClass(ts1, ts2, similarity_ts_config)
            for ts2_name, plot_name, generated_plots in similarity_ts.get_plot_computer():
                __save_figures(ts2_name, plot_name, generated_plots, save_directory_path)


def get_best_sample_names(epoch_directory, n_best, window_selection_metric):
    with open(f'{epoch_directory}/../results.json', 'r', encoding='utf-8') as results_file:
        epoch_directory_results = json.load(results_file)
    return get_top_n_files(epoch_directory_results['Individual'], window_selection_metric, n_best)

def get_top_n_files(dictionary, parameter, n):
    sorted_files = sorted(dictionary.items(), key=lambda x: x[1][parameter]['Multivariate'])
    top_n_files = sorted_files[:n]
    file_names = [file[0] for file in top_n_files]
    return file_names


def __save_figures(filename, plot_name, generated_plots, path='results/figures'):
    for plot in generated_plots:
        try:
            dir_path = __create_directory('/'.join(filename.split('/')[1:]), f'{path}/figures', plot_name)
            plot[0].savefig(f'{dir_path}{plot[0].axes[0].get_title()}.pdf', format='pdf', bbox_inches='tight')
        except FileNotFoundError as file_not_found_error:
            print(f'Could not create the figure in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __create_directory(filename, path, plot_name):
    try:
        if plot_name in PlotFactory.get_instance().figures_requires_all_samples:
            original_filename = '-'.join(os.path.splitext(filename)[0].split('/')[:-2])
            dir_path = f'{path}/{original_filename}/{plot_name}/'
        else:
            parent_filename = '-'.join(os.path.splitext(filename)[0].split('/')[:-2])
            original_filename = os.path.join(parent_filename, os.path.splitext(filename)[0].split('/')[-1])
            dir_path = f'{path}/{original_filename}/{plot_name}/'
        os.makedirs(dir_path, exist_ok=True)
    except FileNotFoundError as file_not_found_error:
        print(f'Could not create the directory in path: {file_not_found_error.filename}. This is probably because the path is too long.')
    return dir_path


def compute_metrics(similarity_ts, save_directory_path):
    metrics_results = {"Aggregated": {}, "Individual": {}}
    metric_computer_iterator = similarity_ts.get_metric_computer()
    metrics_sums = defaultdict(float)
    metrics_counts = defaultdict(int)
    for filename, metric_name, computed_metric in metric_computer_iterator:
        if metrics_results["Individual"].get(filename) is None:
            metrics_results["Individual"][filename] = {}
        metrics_results["Individual"][filename][metric_name] = {}
        metrics_results["Individual"][filename][metric_name] = computed_metric
        metrics_sums[metric_name] += computed_metric['Multivariate']
        metrics_counts[metric_name] += 1
    for metric_name in metrics_sums:
        metrics_results["Aggregated"][metric_name] = metrics_sums[metric_name] / metrics_counts[metric_name]
    __save_metrics(json.dumps(metrics_results, indent=4, ensure_ascii=False).encode('utf-8'),
                   path=f'{save_directory_path}/../')
    return metrics_results


def __save_metrics(computed_metrics, path='results/metrics'):
    try:
        os.makedirs(f'{path}', exist_ok=True)
        with open(f'{path}/results.json', 'w', encoding='utf-8') as file:
            file.write(computed_metrics.decode('utf-8'))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the metrics in path: {file_not_found_error.filename}. This is probably because the path is too long.')

def get_metrics_results(epoch_directory, similarity_ts, recompute_metrics):
    metric_results_by_epoch = {}
    metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = {}
    if not recompute_metrics and os.path.exists(f'{get_epoch_parent_path(epoch_directory)}/results.json'):
        with open(f'{get_epoch_parent_path(epoch_directory)}/results.json', 'r', encoding='utf-8') as results:
            metrics_results = json.load(results)
            if set(similarity_ts.similarity_ts_config.metric_config.metrics).issubset(set(metrics_results['Aggregated'].keys())):
                metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = metrics_results
                print(f'\nEpoch {os.path.basename(get_epoch_parent_path(epoch_directory))} already computed. Skipping...')
            else:
                metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = compute_metrics(similarity_ts, epoch_directory)
    else:
        metric_results_by_epoch[get_epoch_parent_path(epoch_directory)] = compute_metrics(similarity_ts, epoch_directory)
    return metric_results_by_epoch
