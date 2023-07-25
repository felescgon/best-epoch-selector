from helpers.csv_reader import get_experiment_tail, get_root_paths


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ExperimentSelector(metaclass=Singleton):
    def __init__(self, experiments_df, experiment_name_column):
        self.experiments_df = experiments_df
        self.experiments_df['epochs'] = self.experiments_df[experiment_name_column].apply(get_experiment_tail)
        self.experiment_name_column = experiment_name_column
        self.experiment_root_paths = get_root_paths(experiments_df[experiment_name_column])


    def select_n_best_experiments(self, metrics, n_best):
        best_experiments = {}
        cols_to_select = [self.experiment_name_column] + ['epochs'] + metrics
        for experiment_root_path in self.experiment_root_paths:
            best_experiments[experiment_root_path] = {}
            experiments_from_root = self.filter_by_root_path(cols_to_select, experiment_root_path)
            best_epochs_by_metric = self.get_best_epochs_by_metric(metrics, n_best, experiments_from_root)
            best_experiments[experiment_root_path] = list(set.intersection(*best_epochs_by_metric))
        return best_experiments

    def filter_by_root_path(self, cols_to_select, experiment_root_path):
        filtered_experiments = self.experiments_df[cols_to_select]
        experiments_from_root = filtered_experiments[filtered_experiments[self.experiment_name_column].str.contains(experiment_root_path)]
        return experiments_from_root

    def get_best_epochs_by_metric(self, metrics, n_best, experiments_from_root):
        best_epochs_by_metric = []
        for metric in metrics:
            experiments_from_root = experiments_from_root.sort_values(by=metric)
            best_metric_experiments = experiments_from_root.head(n_best)['epochs'].tolist()
            best_epochs_by_metric.append(set(best_metric_experiments))
        return best_epochs_by_metric
