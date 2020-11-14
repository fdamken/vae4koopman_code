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


def compute_energy(name: str, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the energy of the system.

    :param name: The name of the system, used for selecting the appropriate equations.
    :param state: Shape ``(T, p)``, where ``T`` is the number of time steps and ``p`` is the state dimensionality. The state of the system.
    :return: Kinetic and potential energy in that order in a tuple. The total energy is kinetic plus potential energy.
    """

    if name == 'pendulum' or name == 'pendulum_damped':
        m = 1.0
        g = 1.0
        L = 1.0

        theta = state[:, 0]
        theta_dot = state[:, 1]

        T = (m * theta_dot ** 2) / 2.0
        V = m * g * L * np.cos(theta)
    elif name == 'cartpole_gym':
        m_p = 0.1
        m_c = 1.0
        L = 0.5 * 2
        g = 9.81

        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]

        T_cart = (m_c * x_dot ** 2) / 2.0
        T_pole = (m_p * (theta_dot ** 2 + L ** 2 + 2 * np.cos(theta) * theta_dot * x_dot * L + x_dot ** 2)) / 2.0
        T = T_cart + T_pole
        V = -m_p * g * L * np.cos(theta)
    else:
        assert False, f'Unknown name {name}!'
    return T, V


def compute_energies(data: List[Tuple[str, ExperimentConfig, ExperimentResult, List[np.ndarray]]]) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    return_result = []
    for _, config, result, obs_rollouts in data:
        res = []
        for n in range(config.N):
            true_energy = compute_energy(config.name, result.observations[n])
            pred_energy = compute_energy(config.name, obs_rollouts[n])
            res.append((*true_energy, *pred_energy))
        return_result.append(res)
    # noinspection PyTypeChecker
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

    print('Computing energies.')
    runs_energies = compute_energies(data)

    print('Creating plots.')
    for (run_id, config, _, _), energies in zip(data, runs_energies):
        domain = np.arange(config.T) * config.h
        for n, (true_kinetic_energy, true_potential_energy, pred_kinetic_energy, pred_potential_energy) in enumerate(energies):
            with SubplotsAndSave(out_dir, f'energy-R{run_id}-N{n}-total') as (fig, ax):
                ax.plot(domain, true_kinetic_energy + true_potential_energy, color='tuda:blue', label='Truth', zorder=1)
                ax.plot(domain, pred_kinetic_energy + pred_potential_energy, color='tuda:orange', label='Rollout', zorder=2)
                ax.axvline((config.T_train - 1) * config.h, color='tuda:red', ls='dotted', label='Prediction Boundary', zorder=3)
                if config.N > 1:
                    ax.set_title('Total Energy, Sequence %d' % (n + 1))
                else:
                    ax.set_title('Total Energy')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel('Energy')
                ax.legend(loc='lower left')

            with SubplotsAndSave(out_dir, f'energy-R{run_id}-N{n}-kinetic') as (fig, ax):
                ax.plot(domain, true_kinetic_energy, color='tuda:blue', label='Truth', zorder=1)
                ax.plot(domain, pred_kinetic_energy, color='tuda:orange', label='Rollout', zorder=2)
                ax.axvline((config.T_train - 1) * config.h, color='tuda:red', ls='dotted', label='Prediction Boundary', zorder=3)
                if config.N > 1:
                    ax.set_title('Kinetic Energy, Sequence %d' % (n + 1))
                else:
                    ax.set_title('Kinetic Energy')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel('Energy')
                ax.legend(loc='lower left')

            with SubplotsAndSave(out_dir, f'energy-R{run_id}-N{n}-potential') as (fig, ax):
                ax.plot(domain, true_potential_energy, color='tuda:blue', label='Truth', zorder=1)
                ax.plot(domain, pred_potential_energy, color='tuda:orange', label='Rollout', zorder=2)
                ax.axvline((config.T_train - 1) * config.h, color='tuda:red', ls='dotted', label='Prediction Boundary', zorder=3)
                if config.N > 1:
                    ax.set_title('Potential Energy, Sequence %d' % (n + 1))
                else:
                    ax.set_title('Potential Energy')
                ax.set_xlabel(r'$t$')
                ax.set_ylabel('Energy')
                ax.legend(loc='lower left')


if __name__ == '__main__':
    main()
