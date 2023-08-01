import os
import argparse
from datetime import datetime
from datacentertracesdatasets import loadtraces
from similarity_ts.metrics.metric_factory import MetricFactory
from similarity_ts.plots.plot_factory import PlotFactory
from helpers.figure_helpers import generate_figures
from helpers.metric_helpers import compute_metrics
from helpers.reader_utils import get_best_epochs_directories, get_ts2s_directories, load_best_experiments_file


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
        '-recompute',
        '--recompute_metrics',
        help='<Optional> Recompute metrics always.',
        required=False,
        default=False,
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
    save_directory_folder = f'{os.path.abspath(arguments.time_series_2_path)}/results/{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    header_ts1 = tuple(loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300).columns.to_list())
    ts1 = loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300, format='ndarray')
    experiment_directories = get_ts2s_directories(arguments.time_series_2_path)
    compute_metrics(arguments, header_ts1, ts1, experiment_directories, save_directory_folder)
    best_experiments = load_best_experiments_file(save_directory_folder, arguments.n_best)
    if arguments.figures:
        best_epochs_directories = get_best_epochs_directories(best_experiments)
        generate_figures(best_epochs_directories, save_directory_folder, arguments)


if __name__ == '__main__':
    main()
