import sys
from argparse import ArgumentParser

from src.experiment import run_experiment


def main():
    parser = ArgumentParser()
    parser.add_argument('data_file_name')
    parser.add_argument('-e', '--experiment', required=False)
    parser.add_argument('-o', '--output', default='tmp_results_hyperparameter_search')
    args = parser.parse_args()
    data_file_name = args.data_file_name
    experiment = args.experiment
    if not experiment:
        print(f'HyperSearch: Experiment name not explicitly set, using data file name <{data_file_name}>.')
        experiment = data_file_name
    output = args.output

    dry_run_successful = run_experiment(data_file_name, ['with', experiment], results_dir=output, dry_run=True)
    if not dry_run_successful:
        print('HyperSearch: Dry run of experiment setup failed. Aborting.', file=sys.stderr)
        quit(1)
    print('HyperSearch: Dry run of experiment setup successful. Continuing to hyperparameter search.')


if __name__ == '__main__':
    main()
