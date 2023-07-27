import os
import json
from tqdm import tqdm
from collections import defaultdict
from similarity_ts.similarity_ts_config import SimilarityTsConfig
from similarity_ts.metrics.metric_config import MetricConfig
from similarity_ts.plots.plot_config import PlotConfig
from similarity_ts.plots.plot_factory import PlotFactory


def create_similarity_ts_config(arguments, ts2_names, header_names):
    metric_config = None
    plot_config = None if arguments.timestamp_frequency_seconds is None else PlotConfig(None,
                                                                                        arguments.timestamp_frequency_seconds)
    if arguments.metrics is not None or arguments.figures is not None:
        metric_config = MetricConfig(arguments.metrics) if arguments.metrics else MetricConfig([])
        plot_config = PlotConfig(arguments.figures,
                                 arguments.timestamp_frequency_seconds) if arguments.figures else PlotConfig([],
                                                                                                             arguments.timestamp_frequency_seconds)
    return SimilarityTsConfig(metric_config, plot_config, arguments.stride, arguments.window_selection_metric,
                              ts2_names,
                              header_names)


def compute_figures(similarity_ts, save_directory_path):
    plot_computer_iterator = similarity_ts.get_plot_computer()
    tqdm_plot_computer_iterator = tqdm(plot_computer_iterator, total=len(plot_computer_iterator),
                                       desc='Computing plots  ', dynamic_ncols=True)
    for ts2_name, plot_name, generated_plots in tqdm_plot_computer_iterator:
        tqdm_plot_computer_iterator.set_postfix(file=f'{ts2_name}|{plot_name}')
        __save_figures(ts2_name, plot_name, generated_plots, path=save_directory_path)


def __save_figures(filename, plot_name, generated_plots, path='results/figures'):
    for plot in generated_plots:
        try:
            dir_path = __create_directory(filename, f'{path}/figures', plot_name)
            plot[0].savefig(f'{dir_path}{plot[0].axes[0].get_title()}.pdf', format='pdf', bbox_inches='tight')
        except FileNotFoundError as file_not_found_error:
            print(f'Could not create the figure in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __create_directory(filename, path, plot_name):
    try:
        if plot_name in PlotFactory.get_instance().figures_requires_all_samples:
            dir_path = f'{path}/{plot_name}/'
        else:
            original_filename = os.path.splitext(filename)[0]
            dir_path = f'{path}/{original_filename}/{plot_name}/'
        os.makedirs(dir_path, exist_ok=True)
    except FileNotFoundError as file_not_found_error:
        print(f'Could not create the directory in path: {file_not_found_error.filename}. This is probably because the path is too long.')
    return dir_path


def compute_metrics(similarity_ts, save_directory_path):
    metrics_results = {}
    metric_computer_iterator = similarity_ts.get_metric_computer()
    tqdm_metric_computer_iterator = tqdm(metric_computer_iterator, total=len(metric_computer_iterator),
                                         desc='Computing metrics')
    metrics_sums = defaultdict(float)
    metrics_counts = defaultdict(int)
    for _, metric_name, computed_metric in tqdm_metric_computer_iterator:
        metrics_sums[metric_name] += computed_metric['Multivariate']
        metrics_counts[metric_name] += 1
    metrics_results = {}
    for metric_name in metrics_sums:
        metrics_results[metric_name] = metrics_sums[metric_name] / metrics_counts[metric_name]
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
