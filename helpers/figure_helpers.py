import os
import copy
from tqdm import tqdm
from datacentertracesdatasets import loadtraces
from similarity_ts.plots.plot_factory import PlotFactory
from experiments_selection.similarity_copy import SimilarityCopy
from helpers.reader_utils import get_best_sample_names, load_ts_from_path,load_ts_from_csv
from helpers.similarity_ts_utils import create_similarity_ts_config


def compute_figures(best_epochs_directories, save_directory_path, arguments):
    header_ts1 = tuple(loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300).columns.to_list())
    ts1 = loadtraces.get_trace(trace_name=arguments.trace_name, trace_type='machine_usage', stride_seconds=300, format='ndarray')
    tqdm_epoch_iterator = tqdm(best_epochs_directories, total=len(best_epochs_directories), desc='Figures')
    for epoch_directory in tqdm_epoch_iterator:
        tqdm_epoch_iterator.set_postfix(file='/'.join(epoch_directory.split(os.path.sep)[1:-1]))
        ts2_dict = load_ts_from_path(epoch_directory)
        __generate_figures_requires_all_samples(save_directory_path, arguments, header_ts1, ts1, ts2_dict)
        __generate_figures_by_filename(save_directory_path, arguments, header_ts1, ts1, epoch_directory)


def __generate_figures_requires_all_samples(save_directory_path, arguments, header_ts1, ts1, ts2_dict):
    arguments_copy = copy.deepcopy(arguments)
    arguments_copy.figures = [plot for plot in arguments.figures if plot in PlotFactory.get_instance().figures_requires_all_samples]
    similarity_ts_config = create_similarity_ts_config(arguments_copy, list(ts2_dict.keys()), header_ts1)
    similarity_ts = SimilarityCopy(ts1, list(ts2_dict.values()), similarity_ts_config)
    for ts2_name, plot_name, generated_plots in similarity_ts.get_plot_computer():
        __save_figures(ts2_name, plot_name, generated_plots, save_directory_path)


def __generate_figures_by_filename(save_directory_path, arguments, header_ts1, ts1, epoch_directory):
    top_n_files = get_best_sample_names(epoch_directory, arguments.n_best, arguments.window_selection_metric)
    for filename in top_n_files:
        ts2 = load_ts_from_csv(filename, False)
        arguments_copy = copy.deepcopy(arguments)
        arguments_copy.figures = [plot for plot in arguments.figures if plot not in PlotFactory.get_instance().figures_requires_all_samples]
        similarity_ts_config = create_similarity_ts_config(arguments_copy, [filename], header_ts1)
        similarity_ts = SimilarityCopy(ts1, ts2, similarity_ts_config)
        for ts2_name, plot_name, generated_plots in similarity_ts.get_plot_computer():
            __save_figures(ts2_name, plot_name, generated_plots, save_directory_path)


def __save_figures(filename, plot_name, generated_plots, path='results/figures'):
    for plot in generated_plots:
        try:
            dir_path = __create_figures_directory('/'.join(filename.split('/')[1:]), f'{path}/figures', plot_name)
            plot[0].savefig(f'{dir_path}{plot[0].axes[0].get_title()}.pdf', format='pdf', bbox_inches='tight')
        except FileNotFoundError as file_not_found_error:
            print(f'Could not create the figure in path: {file_not_found_error.filename}. This is probably because the path is too long.')


def __create_figures_directory(filename, path, plot_name):
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
