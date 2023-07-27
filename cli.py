import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from datacentertracesdatasets import loadtraces
from similarity_ts.metrics.metric_factory import MetricFactory
from similarity_ts.plots.plot_factory import PlotFactory
from similarity_ts.similarity_ts import SimilarityTs
from experiments_selection.experiment_selector import ExperimentSelector
from helpers.csv_reader import load_ts_from_path
from helpers.similarity_ts_utils import compute_metrics, create_similarity_ts_config


def main():
    available_metrics = MetricFactory.find_available_metrics().keys()
    available_figures = PlotFactory.find_available_figures().keys()
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        '-t',
        '--trace_name',
        help='<Required> Include the trace name from datacentertracesdatasets.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-ts2',
        '--time_series_2_path',
        help='<Required> Include the path to a csv file or a directory with csv files, each one representing time series.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-m',
        '--metrics',
        nargs='+',
        help='<Optional> Include metrics to be computed as a list separated by spaces.',
        choices=available_metrics,
        required=False,
    )
    parser.add_argument(
        '-f',
        '--figures',
        nargs='+',
        help='<Optional> Include figure names to be generated as a list separated by spaces.',
        choices=available_figures,
        required=False,
    )
    parser.add_argument(
        '-head',
        '--header',
        help='<Optional> If the time-series includes a header row.',
        required=False,
        action='store_true',
    )
    parser.add_argument(
        '-ts_freq_secs',
        '--timestamp_frequency_seconds',
        help='<Optional> Include the frequency in seconds in which samples were taken.',
        required=False,
        default=300,
        type=int,
    )
    parser.add_argument(
        '-strd',
        '--stride',
        help='<Optional> Include the stride to be used in moving windows over samples.',
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        '-w_select_met',
        '--window_selection_metric',
        help='<Optional> Include the chosen metric used to pick the best window in the first time series.',
        required=False,
        default='dtw',
        type=str,
    )
    parser.add_argument(
        '-n',
        '--n_best',
        help='<Optional> Require the number of best experiments to compare between metrics.',
        type=int,
        required=False,
        default=4
    )
    parser.add_argument(
        '-m_to_comp',
        '--metrics_to_compare',
        nargs='+',
        help='<Required> Include the metrics to be analysed.',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    __main_script(args)


def __main_script(arguments):
    header_ts1 = tuple(loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300).columns.to_list())
    ts1 = loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300, format='ndarray')
    epoch_directories = get_ts2s_directories(arguments.time_series_2_path)
    metric_results_by_epoch = {}
    for epoch_directory in tqdm(epoch_directories, total=len(epoch_directories), desc='Epochs'):
        tqdm.write(f'Epoch: {epoch_directory}')
        ts2_dict = load_ts_from_path(epoch_directory)
        if len(ts2_dict.values()) != 0:
            similarity_ts_config = create_similarity_ts_config(arguments, list(ts2_dict.keys()), header_ts1)
            similarity_ts = SimilarityTs(ts1, list(ts2_dict.values()), similarity_ts_config)
            if similarity_ts_config.metric_config.metrics:
                metric_results_by_epoch[epoch_directory] = {}
                metric_results_by_epoch[epoch_directory] = compute_metrics(similarity_ts, epoch_directory)
        else:
            print(f'No time series found in {epoch_directory}.')
    save_directory_folder = f'results/{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    experiment_selector = ExperimentSelector(metric_results_by_epoch, 'experiment_dir_name')
    best_experiments = experiment_selector.select_best_experiments(arguments.metrics_to_compare, arguments.n_best)
    __save_selected_experiments(save_directory_folder, best_experiments, arguments.n_best)


def get_ts2s_directories(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f' Path {path} does not exist.')
    if os.path.isfile(path):
        raise ValueError('Path must be a directory.')
    if os.path.isdir(path):
        epoch_directories = []
        for root, _, _ in os.walk(path):
            if root.endswith('generated_data'):
                epoch_directories.append(fix_directory(root))
    return epoch_directories

def fix_directory(root_path):
    path_components = os.path.normpath(root_path).split(os.sep)
    new_root_path = os.sep.join(path_components[:-1])
    return new_root_path.replace(os.path.sep, '/')

def __save_selected_experiments(save_directory_path, experiments, n_best):
    try:
        experiments = json.dumps(experiments, indent=4, ensure_ascii=False).encode('utf-8')
        os.makedirs(f'{save_directory_path}', exist_ok=True)
        with open(f'{save_directory_path}/{n_best}_epochs_by_experiment.json', 'w', encoding='utf-8') as file:
            file.write(experiments.decode('utf-8'))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the best experiments in path: {file_not_found_error.filename}. This is probably because the path is too long.')


if __name__ == '__main__':
    main()
