import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import permutations, product
from similarity_ts.metrics.metric_factory import MetricFactory
from sklearn.feature_selection import mutual_info_regression
from helpers.reader_utils import get_every_experiment_df
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


def generate_features_analysis(arguments, save_directory_folder):
    print('Generating features analysis...')
    dtypes = get_dtypes(arguments.features_to_analyse)
    experiments_df = get_every_experiment_df(save_directory_folder, arguments.features_to_analyse, dtypes)
    save_directory_folder = f'{save_directory_folder}/features_analysis'
    os.makedirs(save_directory_folder, exist_ok=True)
    separate_features = get_separate_features_dict(arguments.features_to_analyse, experiments_df)
    assert len(separate_features['metrics']) > 1 and len(separate_features['hyperparameters']) > 0, 'Feature analysis requires at least 2 metrics and 1 hyperparameter.'
    save_correlation_heatmap(experiments_df, save_directory_folder)
    save_correlation_dendogram(experiments_df, save_directory_folder)
    save_regression_plots(experiments_df, separate_features, save_directory_folder)
    save_histogram_plots(experiments_df, separate_features, save_directory_folder)
    save_scatter_by_category_plots(experiments_df, separate_features, save_directory_folder)
    save_category_plots(experiments_df, separate_features, save_directory_folder)
    save_mutual_information(experiments_df, separate_features, save_directory_folder)


def get_dtypes(features_to_analyse):
    all_metrics = list(MetricFactory.find_available_metrics().keys())
    return {feature: 'category' if feature not in all_metrics else float for feature in features_to_analyse}


def get_separate_features_dict(features_to_analyse, experiments_df):
    all_metrics = list(MetricFactory.find_available_metrics().keys())
    included_metrics = [feature for feature in features_to_analyse if feature in all_metrics]
    included_hyperparameters = [feature for feature in features_to_analyse if feature not in all_metrics and experiments_df[feature].nunique() <= 9]
    return {'metrics': included_metrics, 'hyperparameters': included_hyperparameters}


def get_regression_plot_arguments(separate_features):
    metric_permutations = list(permutations(separate_features['metrics'], 2))
    hyperparameter_combinations = [item for item in separate_features['hyperparameters'] for _ in range(len(metric_permutations))]
    plot_arguments = {
        'axes': metric_permutations * len(separate_features['hyperparameters']),
        'hue': hyperparameter_combinations
    }
    return plot_arguments


def get_histogram_plot_arguments(separate_features):
    return list(product(separate_features['metrics'], separate_features['hyperparameters']))


def get_scatter_by_category_plot_arguments(separate_features):
    hyperparameter_permutations = list(permutations(separate_features['hyperparameters'], 2))
    metric_combinations = [item for item in separate_features['metrics'] for _ in range(len(hyperparameter_permutations))]
    plot_arguments = {
        'hue': hyperparameter_permutations * len(separate_features['metrics']),
        'metric': metric_combinations
    }
    return plot_arguments


def save_correlation_heatmap(experiments_df, save_directory_folder):
    experiments_df = experiments_df.loc[:, experiments_df.nunique() > 1]
    assert not experiments_df.empty, "No valid data left after dropping columns with no variance."
    plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(experiments_df.corr(), dtype=bool))
    heatmap = sns.heatmap(experiments_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    os.makedirs(f'{save_directory_folder}/correlation', exist_ok=True)
    plt.savefig(f'{save_directory_folder}/correlation/triangle_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(experiments_df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    plt.savefig(f'{save_directory_folder}/correlation/matrix_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def save_correlation_dendogram(experiments_df, save_directory_folder):
    experiments_df = experiments_df.loc[:, experiments_df.nunique() > 1]
    assert not experiments_df.empty, "No valid data left after dropping columns with no variance."
    numerical_cols = experiments_df.select_dtypes("float").columns
    corrmat = experiments_df[numerical_cols].corr(method='spearman')
    cg = sns.clustermap(corrmat, linewidths=0.1, annot=True)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    os.makedirs(f'{save_directory_folder}/correlation', exist_ok=True)
    plt.savefig(f'{save_directory_folder}/correlation/dendogram.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def save_regression_plots(experiments_df, separate_features, save_directory_folder):
    plot_arguments = get_regression_plot_arguments(separate_features)
    os.makedirs(f'{save_directory_folder}/regression', exist_ok=True)
    for axes, hue in zip(plot_arguments['axes'], plot_arguments['hue']):
        g = sns.FacetGrid(experiments_df, col=hue, col_wrap=min(experiments_df[hue].nunique(), 3), height=4)
        g.map(sns.regplot, axes[0], axes[1], scatter_kws={'s': 5}, line_kws={'color': 'orange'})
        os.makedirs(f'{save_directory_folder}/regression/regplot/{hue}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/regression/regplot/{hue}/x_{axes[0]}_y_{axes[1]}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        sns.lmplot(x=axes[0], y=axes[1], data=experiments_df, hue=hue, scatter_kws={'s': 5})
        os.makedirs(f'{save_directory_folder}/regression/lmplot/{hue}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/regression/lmplot/{hue}/x_{axes[0]}_y_{axes[1]}.pdf', format='pdf', bbox_inches='tight')
        plt.close()


def save_histogram_plots(experiments_df, separate_features, save_directory_folder):
    plot_arguments = get_histogram_plot_arguments(separate_features)
    os.makedirs(f'{save_directory_folder}/histogram', exist_ok=True)
    for x, hue in plot_arguments:
        sns.kdeplot(data=experiments_df, x=x, hue=hue, fill=True)
        os.makedirs(f'{save_directory_folder}/histogram/{hue}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/histogram/{hue}/x_{x}.pdf', format='pdf', bbox_inches='tight')
        plt.close()


def save_scatter_by_category_plots(experiments_df, separate_features, save_directory_folder):
    os.makedirs(f'{save_directory_folder}/scatter_by_category', exist_ok=True)
    plot_arguments = get_scatter_by_category_plot_arguments(separate_features)
    for hue, metric in zip(plot_arguments['hue'], plot_arguments['metric']):
        sns.set(style="whitegrid")
        sns.catplot(x=hue[0], y=metric, data=experiments_df, hue=hue[1], s=5, kind="strip", col=hue[1], col_wrap=min(experiments_df[hue[1]].nunique(), 3), height=4, aspect=1.2)
        os.makedirs(f'{save_directory_folder}/scatter_by_category/{hue[0]}_vs_{hue[1]}', exist_ok=True)
        plt.savefig(f'{save_directory_folder}/scatter_by_category/{hue[0]}_vs_{hue[1]}/y_{metric}.pdf', format='pdf', bbox_inches='tight')
        plt.close()


def save_category_plots(experiments_df, separate_features, save_directory_folder):
    plot_arguments = get_scatter_by_category_plot_arguments(separate_features)
    for kind in ["boxen","violin","bar"]:
        os.makedirs(f'{save_directory_folder}/{kind}', exist_ok=True)
        for hue, metric in zip(plot_arguments['hue'], plot_arguments['metric']):
            sns.catplot(x=hue[0], y=metric, data=experiments_df, kind=kind, hue=hue[1], col=hue[1])
            os.makedirs(f'{save_directory_folder}/{kind}/{hue[0]}_vs_{hue[1]}', exist_ok=True)
            plt.savefig(f'{save_directory_folder}/{kind}/{hue[0]}_vs_{hue[1]}/y_{metric}.pdf', format='pdf', bbox_inches='tight')
            plt.close()


def make_mi_scores(experiments_df, target_metric, discrete_features):
        mi_scores = mutual_info_regression(experiments_df, target_metric, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=experiments_df.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores.to_dict()


def save_mutual_information(experiments_df, separate_features, save_directory_folder):
    mutual_information_by_metric = {}
    for metric in separate_features['metrics']:
        experiments_df_wo_metric = experiments_df.copy()
        target_metric = experiments_df_wo_metric.pop(metric)
        for feature in experiments_df_wo_metric.select_dtypes("category"):
            experiments_df_wo_metric[feature],_ = experiments_df_wo_metric[feature].factorize()
        discrete_features = experiments_df_wo_metric.dtypes == int
        mi_scores = make_mi_scores(experiments_df_wo_metric, target_metric, discrete_features)
        mutual_information_by_metric[metric] = mi_scores
    mutual_info_folder = f'{save_directory_folder}/mutual_information'
    os.makedirs(mutual_info_folder, exist_ok=True)
    with open(f'{mutual_info_folder}/mutual_information.json', 'w') as f:
        json.dump(mutual_information_by_metric, f, indent=4, ensure_ascii=False)
