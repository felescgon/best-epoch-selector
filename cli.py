import os
import json
import argparse
from experiments_selection.experiment_selector import ExperimentSelector
from helpers.csv_reader import load_df_from_csv


def main():
    parser = argparse.ArgumentParser(
        usage='main -f path_to_file -exp_col experiment_name [-m] [js ...] -n 50'
    )
    parser.add_argument(
        '-f',
        '--filepath',
        help='<Required> Include the path to a csv file.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-exp_name_col',
        '--experiment_name_column',
        help='<Required> Include the name of the column that contains the experiment names.',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-n',
        '--n_best',
        help='<Optional> Require the number of best experiments to compare between metrics.',
        type=int,
        required=False,
        default=50
    )
    parser.add_argument(
        '-m',
        '--metrics',
        nargs='+',
        help='<Required> Include the metrics to be analysed.',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    __main_script(args)


def __main_script(arguments):
    experiments = load_df_from_csv(arguments.filepath)
    experiment_selector = ExperimentSelector(experiments, arguments.experiment_name_column)
    best_experiments = experiment_selector.select_n_best_experiments(arguments.metrics, arguments.n_best)
    __save_selected_experiments('results', best_experiments)

def __save_selected_experiments(path, experiments):
    try:
        experiments = json.dumps(experiments, indent=4, ensure_ascii=False).encode('utf-8')
        os.makedirs(f'{path}', exist_ok=True)
        with open(f'{path}/results.json', 'w', encoding='utf-8') as file:
            file.write(experiments.decode('utf-8'))
    except FileNotFoundError as file_not_found_error:
        print(f'Could not store the best experiments in path: {file_not_found_error.filename}. This is probably because the path is too long.')


if __name__ == '__main__':
    main()
