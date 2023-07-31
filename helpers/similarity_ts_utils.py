from similarity_ts.similarity_ts_config import SimilarityTsConfig
from similarity_ts.metrics.metric_config import MetricConfig
from similarity_ts.plots.plot_config import PlotConfig


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
