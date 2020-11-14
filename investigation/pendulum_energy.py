import os
import shutil
import sys
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from progressbar import progressbar, Percentage, Bar, ETA

from investigation.plot_util import SubplotsAndSave
from investigation.util import load_run, NoResultsFoundException, ExperimentResult, ExperimentConfig
from src.rollout import compute_rollout

LENGTH = 1
GRAVITY = 1
MASS = 1

DIM_POSITION = 0
DIM_VELOCITY = 1


def compute_energy(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    T = (MASS * velocity ** 2) / 2.0
    V = MASS * GRAVITY * LENGTH * np.cos(position)
    return T + V


def compute_energies(data: List[Tuple[str, ExperimentConfig, ExperimentResult, List[np.ndarray]]]) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    return_result = []
    for _, config, result, obs_rollouts in data:
        res = []
        for n in range(config.N):
            true_energy = compute_energy(result.observations[n, :, DIM_POSITION], result.observations[n, :, DIM_VELOCITY])
            pred_energy = compute_energy(obs_rollouts[n][:, DIM_POSITION], obs_rollouts[n][:, DIM_VELOCITY])
            res.append((true_energy, pred_energy))
        return_result.append(res)
    return return_result


def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='investigation/tmp_figures')
    parser.add_argument('-d', '--result_dir', required=True)
    parser.add_argument('-r', '--runs', required=True)
    args = parser.parse_args()
    out_dir = args.out_dir
    result_dir = args.result_dir
    run_ids = args.runs.split(',')

    bar = progressbar.ProgressBar(widgets=['        Loading Runs: ', Percentage(), ' ', Bar(), ' ', ETA()], maxval=len(run_ids)).start()
    runs = []
    for i, run_id in enumerate(run_ids):
        try:
            runs.append((run_id, *load_run(f'{result_dir}/{run_id}', 'run', 'metrics')))
        except FileNotFoundError:
            print(f'No run found for id {run_id}! Ignoring.', file=sys.stderr)
        except NoResultsFoundException:
            print(f'No results found for run {run_id}! Ignoring.', file=sys.stderr)
        bar.update(i + 1)
    bar.finish()

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    bar = progressbar.ProgressBar(widgets=['Calculating Rollouts: ', Percentage(), ' ', Bar(), ' ', ETA()], maxval=len(runs)).start()
    data = []
    for i, (run_id, config, result, _) in enumerate(runs):
        _, (obs_rollouts, _), _ = compute_rollout(config, result, config.N)
        data.append((run_id, config, result, obs_rollouts))
        bar.update(i + 1)
    bar.finish()

    runs_energies = compute_energies(data)
    for (run_id, config, _, _), energies in zip(data, runs_energies):
        domain = np.arange(config.T) * config.h
        for n, (true_energy, pred_energy) in enumerate(energies):
            with SubplotsAndSave(out_dir, f'energy-R{run_id}-N{n}') as (fig, ax):
                ax.plot(domain, true_energy, label='True Energy')
                ax.plot(domain, pred_energy, label='Rollout Energy')
                if config.N > 1:
                    ax.set_title('Sequence %d' % (n + 1))
                ax.set_xlabel(r'$t$')
                ax.set_ylabel('Energy')
                ax.legend(loc='lower left')


if __name__ == '__main__':
    main()
