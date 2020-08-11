from typing import List

import numpy as np

from investigation.plot_util import SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentResult, load_run



def compute_rollout(config: ExperimentConfig, result: ExperimentResult) -> np.ndarray:
    rollout = np.zeros((config.T, config.latent_dim))
    rollout[0, :] = result.m0
    for t in range(1, config.T):
        rollout[t, :] = result.A @ rollout[t - 1, :]
    return rollout



def computer_observations(result: ExperimentResult, latent_trajectory: np.ndarray) -> np.ndarray:
    return result.g_numpy(latent_trajectory)



def plot_latent_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, latent_trajectories: List[np.ndarray]):
    domain = np.arange(config.T) * config.h

    with SubplotsAndSave(out_dir, 'rollout-latents', config.N, config.latent_dim,
                         sharex = 'all',
                         sharey = 'row',
                         figsize = (2 + 5 * config.latent_dim, 1 + 4 * config.N),
                         squeeze = False) as (fig, axss):
        for n, (axs, latent_trajectory) in enumerate(zip(axss, latent_trajectories)):
            for dim, ax in enumerate(axs):
                ax.plot(domain, result.estimations_latents[n, dim, :], label = 'Filtered/Smoothed')
                ax.plot(domain, latent_trajectory[:, dim], label = 'Rollout')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Latents')
                ax.legend()
        fig.tight_layout()



def plot_observations_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, observation_trajectories: List[np.ndarray]):
    domain = np.arange(config.T) * config.h

    with SubplotsAndSave(out_dir, 'rollout-observations', config.N, config.observation_dim,
                         sharex = 'all',
                         sharey = 'row',
                         figsize = (2 + 5 * config.observation_dim, 1 + 4 * config.N),
                         squeeze = False) as (fig, axss):
        for n, (axs, observation_trajectory) in enumerate(zip(axss, observation_trajectories)):
            for dim, ax in enumerate(axs):
                ax.plot(domain, result.observations[n, :, dim], label = 'Filtered/Smoothed')
                ax.plot(domain, observation_trajectory[:, dim], label = 'Rollout')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observations')
                ax.legend()
        fig.tight_layout()



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, _ = load_run('tmp_results/248', 'checkpoint_00020', None)

    latent_trajectories = []
    observation_trajectories = []
    for _ in range(config.N):
        latent_trajectory = compute_rollout(config, result)
        observation_trajectory = computer_observations(result, latent_trajectory)
        latent_trajectories.append(latent_trajectory)
        observation_trajectories.append(observation_trajectory)

    plot_latent_rollout(config, result, out_dir, latent_trajectories)
    plot_observations_rollout(config, result, out_dir, observation_trajectories)
