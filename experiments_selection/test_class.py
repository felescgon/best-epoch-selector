import concurrent.futures
from tqdm import tqdm
from similarity_ts.similarity_ts import SimilarityTs
from similarity_ts.similarity_ts_config import SimilarityTsConfig
from similarity_ts.metrics.metric_factory import MetricFactory
from similarity_ts.plots.plot_factory import PlotFactory
from similarity_ts.helpers.window_sampler import split_ts_strided

class TestClass(SimilarityTs):
    def __init__(self, ts1, ts2s, similarity_ts_config=None):
        self.ts1 = ts1
        self.ts2s = ts2s
        self.similarity_ts_config = similarity_ts_config if similarity_ts_config is not None else SimilarityTsConfig()
        self.header_names = self.similarity_ts_config.header_names if self.similarity_ts_config.header_names is not None else [
            'column-' + str(i)
            for i in
            range(ts1.shape[1])]
        self.ts2_dict = self.__build_ts2_dict(self.ts2s, self.similarity_ts_config.ts2_names)
        self.ts1_windows = split_ts_strided(self.ts1, self.ts2s[0].shape[0], self.similarity_ts_config.stride)
        self.ts1_ts2_associated_windows = self.__create_ts1_ts2_associated_windows()
        self.metric_factory = MetricFactory.get_instance(self.similarity_ts_config.metric_config.metrics)
        self.plot_factory = PlotFactory.get_instance(self.similarity_ts_config.plot_config.figures)


    def __build_ts2_dict(self, ts2s, ts2_filenames):
        ts2_filenames = ts2_filenames if ts2_filenames is not None else ['ts2_' + str(i) for i in range(len(ts2s))]
        return {ts2_name: ts2 for ts2, ts2_name in zip(ts2s, ts2_filenames)}
    


    def __process_single_ts2_item(self, item):
        filename, ts2 = item
        metric_object = MetricFactory.get_metric_by_name(self.similarity_ts_config.window_selection_metric)
        most_similar_ts1_sample = self.__get_most_similar_ts_sample(self.ts1_windows, ts2, metric_object)
        cached_metric = {self.similarity_ts_config.window_selection_metric: metric_object.compute(most_similar_ts1_sample, ts2, self)}
        return filename, {
            'most_similar_ts1_sample': most_similar_ts1_sample,
            'ts2': ts2,
            'cached_metric': cached_metric
        }

    def __create_ts1_ts2_associated_windows(self):
        ts1_ts2_associated_windows = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:  # Puedes cambiar ProcessPoolExecutor por ThreadPoolExecutor si prefieres hilos en lugar de procesos
            items = self.ts2_dict.items()
            results = list(tqdm(executor.map(self.__process_single_ts2_item, items), total=len(self.ts2_dict), desc='Selecting most similar windows'))
        
        for filename, result in results:
            ts1_ts2_associated_windows[filename] = result
        
        return ts1_ts2_associated_windows



    def __get_most_similar_ts_sample(self, ts1_windows, ts2, metric_object):
        current_best = float('inf')
        most_similar_sample = []
        for ts1_window in ts1_windows:
            current_distance = metric_object.compute_distance(ts1_window, ts2)
            if metric_object.compare(current_distance, current_best) > 0:
                current_best = current_distance
                most_similar_sample = ts1_window
        return most_similar_sample
