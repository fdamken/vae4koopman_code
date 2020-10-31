import os
import shutil
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from matplotlib import ticker

from investigation.plot_util import SubplotsAndSave, figsize, even, tuda
from investigation.rollout import compute_rollout
from investigation.util import load_run, ExperimentMetrics, ExperimentResult, ExperimentConfig


def calculate_metric(result: ExperimentResult, n: int, obs_rollout: np.ndarray, metric_name: str) -> float:
    if metric_name == 'rmse':
        expected = result.observations[n]
        actual = obs_rollout
        return np.sqrt(((expected - actual) ** 2).mean())
    else:
        assert False


def calculate_metrics(runs: List[Tuple[ExperimentConfig, ExperimentResult, ExperimentMetrics]], metric_name: str, accumulation_method: str) -> np.ndarray:
    res = []
    for (config, result, metrics) in runs:
        _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
        metrics = []
        for n in range(config.N):
            metrics.append(calculate_metric(result, n, obs_rollouts[n], metric_name))
        if accumulation_method == 'mean':
            metric = np.mean(metrics)
        elif accumulation_method == 'first':
            metric = metrics[0]
        else:
            assert False
        res.append(metric)
    return np.asarray(res)


def make_title(metric_name: str, accumulation_method: str) -> str:
    if metric_name == 'rmse':
        label = 'RMSE'
    else:
        assert False
    if accumulation_method == 'mean':
        label += ', Mean over Sequences'
    elif accumulation_method == 'first':
        label += ', First Sequence'
    else:
        assert False
    return label


def make_xlabel(ordinate: str) -> str:
    if ordinate == 'N':
        return 'Number of Training Sequences'
    elif ordinate == 'T_train':
        return 'Training Sequence Length'
    elif ordinate == 'latent_dim':
        return 'Latent Dimensionality'
    assert False


def make_ylabel(metric_name: str) -> str:
    if metric_name == 'rmse':
        return 'RMSE'
    assert False


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='investigation/tmp_figures')
    parser.add_argument('-d', '--result_dir', required=True)
    parser.add_argument('-f', '--from', required=True, type=int, dest='run_from')
    parser.add_argument('-t', '--to', required=False, type=int, dest='run_to')
    parser.add_argument('-m', '--metric', default='rmse')
    parser.add_argument('-a', '--accumulation', default='mean')
    parser.add_argument('-x', '--ordinate')
    args = parser.parse_args()
    out_dir = args.out_dir
    result_dir = args.result_dir
    run_from = args.run_from
    run_to = args.run_to
    metric_name = args.metric.lower()
    accumulation_method = args.accumulation.lower()
    ordinate = args.ordinate

    run_ids = [str(x) for x in range(run_from, run_to + 1)]

    print('Reading results from %s/{%s}.' % (result_dir, ','.join(run_ids)))

    runs = [load_run(result_dir + '/' + run, 'run', 'metrics') for run in run_ids]

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    X = [run[0].config_dict[ordinate] for run in runs]
    Y = calculate_metrics(runs, metric_name, accumulation_method)
    x_data = []
    y_data = []
    for x, y in zip(X, Y):
        if x in x_data:
            y_data[x_data.index(x)].append(y)
        else:
            x_data.append(x)
            y_data.append([y])
    x = np.asarray(x_data)
    y_mean = np.asarray([np.mean(part) for part in y_data])
    y_std = np.asarray([np.std(part) for part in y_data])
    x_ticker_n = max(1, even(len(runs) / 10))
    with SubplotsAndSave(out_dir, 'comparison', 1, 1, figsize=figsize(1, 1)) as (fig, ax):
        ax.plot(x, y_mean, color=tuda('blue'), zorder=1)
        ax.fill_between(x, y_mean - 2 * y_std, y_mean + 2 * y_std, color=tuda('blue'), alpha=0.2, zorder=1)
        ax.scatter(X, Y, s=1, color=tuda('black'), zorder=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticker_n))
        if x_ticker_n > 1:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_ticker_n // 2))
        else:
            ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_title(make_title(metric_name, accumulation_method))
        ax.set_xlabel(make_xlabel(ordinate))
        ax.set_ylabel(make_ylabel(metric_name))
