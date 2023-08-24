import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import permutations, product
from similarity_ts.metrics.metric_factory import MetricFactory
from helpers.reader_utils import get_every_experiment_df
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


def generate_features_analysis(arguments, save_directory_folder):
    print('Generating features analysis...')
    dtypes = get_dtypes(arguments.features_to_analyse)
    experiments_df = get_every_experiment_df(save_directory_folder, arguments.features_to_analyse, dtypes)
    save_directory_folder = f'{save_directory_folder}/features_analysis'
    os.makedirs(save_directory_folder, exist_ok=True)
    separate_features = get_separate_features_dict(arguments.features_to_analyse)
    assert len(separate_features['metrics']) > 1 and len(separate_features['hyperparameters']) > 0, 'Feature analysis requires at least 2 metrics and 1 hyperparameter.'
    plot_arguments = get_plot_arguments(experiments_df, separate_features)
    save_correlation_heatmap(experiments_df, save_directory_folder)
    save_regression_plots(experiments_df, plot_arguments, save_directory_folder)


def get_dtypes(features_to_analyse):
    all_metrics = list(MetricFactory.find_available_metrics().keys())
    return {feature: 'category' if feature not in all_metrics else float for feature in features_to_analyse}


def get_separate_features_dict(features_to_analyse):
    all_metrics = list(MetricFactory.find_available_metrics().keys())
    included_metrics = [feature for feature in features_to_analyse if feature in all_metrics]
    included_hyperparameters = [feature for feature in features_to_analyse if feature not in all_metrics]
    return {'metrics': included_metrics, 'hyperparameters': included_hyperparameters}


def get_plot_arguments(experiments_df, separate_features):
    separate_features['hyperparameters'] = [feature for feature in separate_features['hyperparameters'] if experiments_df[feature].nunique() <= 9]
    metric_permutations = list(permutations(separate_features['metrics'], 2))
    hyperparameter_combinations = [item for item in separate_features['hyperparameters'] for _ in range(len(metric_permutations))]
    plot_arguments = {
        'axes': metric_permutations * len(separate_features['hyperparameters']),
        'hue': hyperparameter_combinations
    }
    return plot_arguments


def save_correlation_heatmap(experiments_df, save_directory_folder):
    plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(experiments_df.corr(), dtype=bool))
    heatmap = sns.heatmap(experiments_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    os.makedirs(f'{save_directory_folder}/correlation', exist_ok=True)
    plt.savefig(f'{save_directory_folder}/correlation/heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def save_regression_plots(experiments_df, plot_arguments, save_directory_folder):
    os.makedirs(f'{save_directory_folder}/regression', exist_ok=True)
    for axes, hue in zip(plot_arguments['axes'], plot_arguments['hue']):
        g = sns.FacetGrid(experiments_df, col=hue, col_wrap=min(len(experiments_df[hue]), 3), height=4)
        g.map(sns.regplot, axes[0], axes[1], scatter_kws={'s': 5}, line_kws={'color': 'orange'})
        os.makedirs(f'{save_directory_folder}/regression/regplot/{hue}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/regression/regplot/{hue}/x_{axes[0]}_y_{axes[1]}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        sns.lmplot(x=axes[0], y=axes[1], data=experiments_df, hue=hue, scatter_kws={'s': 5})
        os.makedirs(f'{save_directory_folder}/regression/lmplot/{hue}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/regression/lmplot/{hue}/x_{axes[0]}_y_{axes[1]}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
