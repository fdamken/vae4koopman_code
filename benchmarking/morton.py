import os
import shutil
from argparse import ArgumentParser
from typing import List

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

from investigation.plot_util import SubplotsAndSave
from investigation.util import load_run, ExperimentConfig, ExperimentResult
from src.rollout import compute_rollout

jsonpickle_numpy.register_handlers()


def compute_nrmse(expected: np.ndarray, actual: np.ndarray) -> float:
    normalization_factor = np.amax(expected, axis=0) - np.amin(expected, axis=0)
    return (np.sqrt(((expected - actual) ** 2).mean(axis=0)) / normalization_factor).mean().item()


def compute_pendulum_nrmse(run: str) -> None:
    config, result, _ = load_run(run, 'run', 'metrics')
    expected = result.observations[0]
    _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
    actual = obs_rollouts[0]
    print('Pendulum (Angle):', compute_nrmse(expected, actual))

    expected_cosine = np.cos(expected[:, 0])
    expected_sine = np.sin(expected[:, 0])
    expected_xy = np.concatenate([expected_cosine, expected_sine, expected[:, 1]], axis=0)
    actual_cosine = np.cos(actual[:, 0])
    actual_sine = np.sin(actual[:, 0])
    actual_xy = np.concatenate([actual_cosine, actual_sine, actual[:, 1]], axis=0)
    print('Pendulum (x/y):  ', compute_nrmse(expected_xy, actual_xy))


def compute_our_nrmse(config: ExperimentConfig, result: ExperimentResult) -> float:
    _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
    return compute_nrmse(result.observations[0], obs_rollouts[0])


def compute_morton_nrmse(file: str) -> float:
    with open(file, 'r') as fh:
        predictions = jsonpickle.loads(fh.read())
    expected = predictions['x'][0, :-1]
    actual = predictions['preds'][predictions['best_pred']]
    return compute_nrmse(expected, actual)


def plot_morton_result(out_dir: str, file: str, observation_dim_names: List[str]) -> None:
    with open(file, 'r') as fh:
        predictions = jsonpickle.loads(fh.read())

    truth = predictions['x'][0, :-1]
    T, observation_dim = truth.shape
    T_train = T // 2 + 1
    pred = predictions['preds']
    domain = np.arange(T)
    pred_mean = np.mean(pred, axis=0)
    pred_min = np.amin(pred, axis=0)
    pred_max = np.amax(pred, axis=0)

    for dim, dim_name in enumerate(observation_dim_names):
        with SubplotsAndSave(out_dir, f'morton-predictions-D{dim}') as (fig, ax):
            plot_morton_result_to_ax(T_train, ax, dim, dim_name, domain, pred_max, pred_mean, pred_min, truth)
            ax.set_xlabel('Time Steps')
    with SubplotsAndSave(out_dir, f'morton-predictions',
                         nrows=int(np.ceil(observation_dim / 2)),
                         ncols=min(observation_dim, 2),
                         sharex='col',
                         squeeze=False) as (fig, axs):
        for dim, (ax, dim_name) in enumerate(zip(axs.flatten(), observation_dim_names)):
            plot_morton_result_to_ax(T_train, ax, dim, dim_name, domain, pred_max, pred_mean, pred_min, truth)
            if dim == observation_dim - 1 or dim == observation_dim - 2:
                ax.set_xlabel('Time Steps')


def plot_morton_result_to_ax(T_train, ax, dim, dim_name, domain, pred_max, pred_mean, pred_min, truth):
    # Ground truth.
    ax.scatter(domain, truth[:, dim], s=1, color='black', label='Truth', zorder=1)
    ax.plot(domain[:T_train], pred_mean[:T_train, dim], color='tuda:blue', label='Estimated Trajectory', zorder=2)
    ax.plot(domain[T_train - 1:], pred_mean[T_train - 1:, dim], ls='dashed', color='tuda:blue', label='Estimated Trajectory (Prediction)', zorder=2)
    ax.fill_between(domain, pred_max[:, dim], pred_min[:, dim], color='tuda:blue', alpha=0.2, label='Min-to-Max Region', zorder=3)
    # Prediction boundary and learned initial value.
    ax.axvline(domain[T_train - 1], color='tuda:red', ls='dotted', label='Prediction Boundary', zorder=1)
    ax.set_ylabel(dim_name)
    ax.legend(loc='upper left').set_zorder(100)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='benchmarking/tmp_figures')
    parser.add_argument('-d', '--result_dir', required=True)
    parser.add_argument('-m', '--morton_result_file', required=True)
    args = parser.parse_args()
    out_dir = args.out_dir
    result_dir = args.result_dir
    morton_result_file = f'{args.morton_result_file}.json'

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    config, result, _ = load_run(result_dir, 'run', 'metrics')

    # compute_pendulum_nrmse('tmp_results_grid_search/cpu/latent-dim_pendulum-cpu/10')
    our_nrmse = compute_our_nrmse(config, result)
    morton_nrmse = compute_morton_nrmse(morton_result_file)
    print('Our Run:    %s' % result_dir)
    print('Morton Run: %s' % morton_result_file)
    print('Our NRMSE:    %8.5f' % our_nrmse)
    print('Morton NRMSE: %8.5f' % morton_nrmse)
    print('Are we better? %s' % ('yes' if our_nrmse < morton_nrmse else 'no'))
    print('Absolute Difference: %8.5f' % np.abs(our_nrmse - morton_nrmse))
    print('Relative Difference: %8.5f' % (max(our_nrmse, morton_nrmse) / min(our_nrmse, morton_nrmse)))

    plot_morton_result(out_dir, morton_result_file, config.observation_dim_names)


if __name__ == '__main__':
    main()
