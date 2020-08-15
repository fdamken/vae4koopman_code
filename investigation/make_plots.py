import os
import shutil
from argparse import ArgumentParser

from investigation.plot_g_final_log_likelihood import plot_g_final_log_likelihood
from investigation.plot_latents import plot_latents
from investigation.plot_log_likelihood import plot_log_likelihood
from investigation.plot_observations import plot_observations
from investigation.util import load_run


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default = 'investigation/tmp_figures')
    parser.add_argument('-d', '--result_dir', required = True)
    parser.add_argument('-f', '--result_file_name', default = 'run')
    parser.add_argument('-m', '--metrics_file_name', required = False)
    parser.add_argument('-i', '--include_plots', required = False)
    args = parser.parse_args()
    out_dir = args.out_dir
    result_dir = args.result_dir
    result_file_name = args.result_file_name
    metrics_file_name = args.metrics_file_name
    include_plots = args.include_plots if args.include_plots is None else ['g_final_log_likelihood', 'latents', 'log_likelihood', 'observations']

    config, result, metrics = load_run(result_dir, result_file_name, metrics_file_name)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    if 'g_final_log_likelihood' in include_plots:
        if metrics is not None:
            plot_g_final_log_likelihood(out_dir, config, result, metrics)
    if 'latents' in include_plots:
        plot_latents(out_dir, config, result)
    if 'log_likelihood' in include_plots:
        if metrics is not None:
            plot_log_likelihood(out_dir, config, result, metrics)
    if 'observations' in include_plots:
        plot_observations(out_dir, config, result)
