import pandas as pd
from helpers.reader_utils import get_experiment_tail, get_root_paths


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BestEpochsSelector(metaclass=Singleton):
    def __init__(self, experiments, experiment_name_column):
        self.experiments_df = self.__dict_to_df(experiments)
        self.experiments_df['epochs'] = self.experiments_df[experiment_name_column].apply(get_experiment_tail)
        self.experiment_name_column = experiment_name_column
        self.experiment_root_paths = get_root_paths(self.experiments_df[experiment_name_column])


    def select_best_epochs(self, metrics, n_best):
        best_experiments = {}
        for experiment_root_path in self.experiment_root_paths:
            best_experiments[experiment_root_path] = {}
            experiments_from_root = self.__filter_by_root_path(experiment_root_path)
            best_experiments[experiment_root_path] = self.__get_best_epochs_from_experiment(metrics, n_best, experiments_from_root)
        return best_experiments


    def __dict_to_df(self, experiments):
        filas = []
        for experiment_dir_name, metrics in experiments.items():
            values_by_metric = {k: v for k, v in metrics['Aggregated'].items()}
            fila = {
                'experiment_dir_name': experiment_dir_name,
            }
            fila.update(values_by_metric)
            filas.append(fila)
        experiments_df = pd.DataFrame(filas)
        return experiments_df


    def __filter_by_root_path(self, experiment_root_path):
        experiments_from_root = self.experiments_df[self.experiments_df[self.experiment_name_column].str.contains(experiment_root_path)]
        return experiments_from_root


    def __get_best_epochs_from_experiment(self, metrics, n_best, experiments):
        experiments_sorted_by_metrics = self.__sort_experiments_by_metrics(metrics, experiments)
        best_epochs_intersection = self.__get_experiment_intersection(n_best, len(experiments), experiments_sorted_by_metrics)
        return best_epochs_intersection


    def __get_experiment_intersection(self, n_best, experiments_length, experiments_sorted_by_metrics):
        left_i = 0
        right_i = experiments_length - 1
        while left_i <= right_i:
            mid_i = (left_i + right_i) // 2
            best_epochs = self.__pick_top_epochs(experiments_sorted_by_metrics, mid_i)
            best_epochs_intersection = {'best_epochs': list(set.intersection(*best_epochs)), 'intersection_experiments_n': mid_i}
            if len(best_epochs_intersection['best_epochs']) == n_best:
                break
            if len(best_epochs_intersection['best_epochs']) < n_best:
                left_i = mid_i + 1
            else:
                right_i = mid_i - 1
        return best_epochs_intersection


    def __pick_top_epochs(self, experiments_sorted_by_metrics, mid_i):
        best_epochs = []
        for sorted_experiments in experiments_sorted_by_metrics:
            best_metric_experiments = sorted_experiments.head(mid_i)['epochs'].tolist()
            best_epochs.append(set(best_metric_experiments))
        return best_epochs


    def __sort_experiments_by_metrics(self, metrics, experiments):
        sorted_experiments = []
        for metric in metrics:
            sorted_experiments.append(experiments.sort_values(by=metric))
        return sorted_experiments
