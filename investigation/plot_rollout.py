from typing import List

import matplotlib.pyplot as plt
import numpy as np

from investigation.util import ExperimentConfig, ExperimentResult, load_run



def compute_rollout(config: ExperimentConfig, result: ExperimentResult) -> np.ndarray:
    rollout = np.zeros((config.T, config.latent_dim))
    rollout[0, :] = result.m0
    for t in range(1, config.T):
        rollout[t, :] = result.A @ rollout[t - 1, :]
    return rollout



def computer_observations(result: ExperimentResult, latent_trajectory: np.ndarray) -> np.ndarray:
    return result.g_numpy(latent_trajectory)



def plot_latent_rollout(config: ExperimentConfig, result: ExperimentResult, latent_trajectories: List[np.ndarray]):
    latent_dim = config.latent_dim
    domain = np.arange(config.T) * config.h

    fig, axss = plt.subplots(config.N, latent_dim, sharex = 'all', sharey = 'row', figsize = (2 + 5 * latent_dim, 4 * config.N), squeeze = False)
    for n, (axs, latent_trajectory) in enumerate(zip(axss, latent_trajectories)):
        for dim, ax in enumerate(axs):
            ax.plot(domain, result.estimations_latents[n, dim, :], label = 'Filtered/Smoothed')
            ax.plot(domain, latent_trajectory[:, dim], label = 'Rollout')
            ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Latents')
            ax.legend()
    fig.suptitle('Filtered/Smoothed and Rollout Latents (%s)' % config.title)
    fig.show()



def plot_observations_rollout(config: ExperimentConfig, result: ExperimentResult, observation_trajectories: List[np.ndarray]):
    observation_dim = config.observation_dim
    domain = np.arange(config.T) * config.h

    fig, axss = plt.subplots(config.N, observation_dim, sharex = 'all', sharey = 'row', figsize = (2 + 5 * observation_dim, 4 * config.N), squeeze = False)
    for n, (axs, observation_trajectory) in enumerate(zip(axss, observation_trajectories)):
        for dim, ax in enumerate(axs):
            ax.plot(domain, result.observations[n, :, dim], label = 'Input')
            ax.plot(domain, observation_trajectory[:, dim], label = 'Rollout')
            ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Observations')
            ax.legend()
    fig.suptitle('Filtered/Smoothed and Rollout Observations (%s)' % config.title)
    fig.show()



if __name__ == '__main__':
    config, result = load_run('tmp_results/138', 'run')

    latent_trajectories = []
    observation_trajectories = []
    for _ in range(config.N):
        latent_trajectory = compute_rollout(config, result)
        observation_trajectory = computer_observations(result, latent_trajectory)
        latent_trajectories.append(latent_trajectory)
        observation_trajectories.append(observation_trajectory)

    plot_latent_rollout(config, result, latent_trajectories)
    plot_observations_rollout(config, result, observation_trajectories)
