import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from similarity_ts.metrics.metric_factory import MetricFactory
from helpers.reader_utils import get_every_experiment_df


def generate_features_analysis(arguments, save_directory_folder):
    dtypes = get_dtypes(arguments.features_to_analyse)
    experiments_df = get_every_experiment_df(save_directory_folder, arguments.features_to_analyse, dtypes)
    save_directory_folder = f'{save_directory_folder}/features_analysis'
    os.makedirs(save_directory_folder)
    save_correlation_heatmap(experiments_df, save_directory_folder)


def get_dtypes(features_to_analyse):
    all_metrics = list(MetricFactory.find_available_metrics().keys())
    return {feature: 'category' if feature not in all_metrics else float for feature in features_to_analyse}


def save_correlation_heatmap(experiments_df, save_directory_folder):
    mask = np.triu(np.ones_like(experiments_df.corr(), dtype=bool))
    heatmap = sns.heatmap(experiments_df.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    os.makedirs(f'{save_directory_folder}/correlation')
    plt.savefig(f'{save_directory_folder}/correlation/heatmap.pdf', format='pdf')
    plt.close()
