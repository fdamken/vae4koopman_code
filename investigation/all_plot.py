import os
import shutil
from argparse import ArgumentParser

from investigation.generate_trajectories import generate_trajectories
from investigation.make_plots import make_plots
from investigation.plot_rollout import plot_rollout
from investigation.predict_trajectories import predict_trajectories
from investigation.util import load_run


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default = 'investigation/tmp_figures')
    parser.add_argument('-d', '--result_dir', required = True)
    parser.add_argument('-f', '--result_file_name', default = 'run')
    parser.add_argument('-m', '--metrics_file_name', required = False)
    args = parser.parse_args()
    out_dir = args.out_dir
    result_dir = args.result_dir
    result_file_name = args.result_file_name
    metrics_file_name = args.metrics_file_name

    config, result, metrics = load_run(result_dir, result_file_name, metrics_file_name)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    generate_trajectories(out_dir, config, result)
    make_plots(out_dir, config, result, metrics)
    plot_rollout(out_dir, config, result)
    predict_trajectories(out_dir, config, result)
