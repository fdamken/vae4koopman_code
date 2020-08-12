from typing import List

import numpy as np

from investigation.plot_util import SubplotsAndSave, tuda
from investigation.util import ExperimentConfig, ExperimentResult, load_run


PREDICTION_STEPS = 750



def compute_rollout(config: ExperimentConfig, result: ExperimentResult, T: int) -> np.ndarray:
    rollout = np.zeros((T, config.latent_dim))
    rollout[0, :] = result.m0
    for t in range(1, T):
        rollout[t, :] = result.A @ rollout[t - 1, :]
    return rollout



def computer_observations(result: ExperimentResult, latent_trajectory: np.ndarray) -> np.ndarray:
    return result.g_numpy(latent_trajectory)



def plot_latent_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, prediction_steps: int, latent_trajectories: List[np.ndarray]):
    T, _ = latent_trajectories[0].shape
    domain = np.arange(T) * config.h
    if prediction_steps > 0:
        domain_before_prediction = domain[:-prediction_steps]
    else:
        domain_before_prediction = domain
    domain_predicted = domain[-prediction_steps:]

    with SubplotsAndSave(out_dir, 'rollout-latents', config.N, config.latent_dim,
                         sharex = 'all',
                         sharey = 'row',
                         figsize = (2 + 5 * config.latent_dim, 1 + 4 * config.N),
                         squeeze = False) as (fig, axss):
        for n, (axs, latent_trajectory) in enumerate(zip(axss, latent_trajectories)):
            if prediction_steps > 0:
                latent_trajectory_before_prediction = latent_trajectory[:-prediction_steps, :]
            else:
                latent_trajectory_before_prediction = latent_trajectory
            latent_trajectory_predicted = latent_trajectory[-prediction_steps:, :]
            for dim, ax in enumerate(axs):
                ax.plot(domain_before_prediction, result.estimations_latents[n, dim, :], color = tuda('orange'), label = 'Filtered/Smoothed')
                ax.plot(domain_before_prediction, latent_trajectory_before_prediction[:, dim], color = tuda('blue'), label = 'Rollout')
                if prediction_steps > 0:
                    ax.plot(domain_predicted, latent_trajectory_predicted[:, dim], color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Latents')
                ax.legend()
        fig.tight_layout()



def plot_observations_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, prediction_steps: int, observation_trajectories: List[np.ndarray]):
    T, _ = observation_trajectories[0].shape
    domain = np.arange(T) * config.h
    if prediction_steps > 0:
        domain_before_prediction = domain[:-prediction_steps]
    else:
        domain_before_prediction = domain
    domain_predicted = domain[-prediction_steps:]

    observation_trajectories_smoothed = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape(
            (config.N, config.T, config.observation_dim))
    with SubplotsAndSave(out_dir, 'rollout-observations', config.N, config.observation_dim,
                         sharex = 'all',
                         figsize = (2 + 5 * config.observation_dim, 1 + 4 * config.N),
                         squeeze = False) as (fig, axss):
        for n, (axs, observation_trajectory, observation_trajectory_smoothed) in enumerate(zip(axss, observation_trajectories, observation_trajectories_smoothed)):
            for dim, ax in enumerate(axs):
                confidence = 2 * np.sqrt(result.R[dim])
                mean = observation_trajectory[:, dim]
                upper = mean + confidence
                lower = mean - confidence
                if prediction_steps > 0:
                    mean_before_prediction = mean[:-prediction_steps]
                else:
                    mean_before_prediction = mean
                mean_predicted = mean[-prediction_steps:]

                ax.scatter(domain_before_prediction, result.observations[n, :, dim], s = 1, color = tuda('black'), label = 'Truth')
                ax.plot(domain_before_prediction, mean_before_prediction, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_predicted, mean_predicted, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                # ax.plot(domain, observation_trajectory_smoothed[:, dim], ls = 'dashdot', label = 'Smoothed/Filtered')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                if prediction_steps > 0:
                    ax.axvline(domain_before_prediction[-1], color = tuda('red'), ls = 'dotted', label = 'Prediction Boundary')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observations')
                ax.legend()
        fig.tight_layout()



def plot_rollout(out_dir, config: ExperimentConfig, result: ExperimentResult):
    latent_trajectories = []
    observation_trajectories = []
    for _ in range(config.N):
        latent_trajectory = compute_rollout(config, result, config.T + PREDICTION_STEPS)
        observation_trajectory = computer_observations(result, latent_trajectory)
        latent_trajectories.append(latent_trajectory)
        observation_trajectories.append(observation_trajectory)

    plot_latent_rollout(config, result, out_dir, PREDICTION_STEPS, latent_trajectories)
    plot_observations_rollout(config, result, out_dir, PREDICTION_STEPS, observation_trajectories)



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, _ = load_run('tmp_results/transferred_results/30', 'checkpoint_00035', 'metrics')

    plot_rollout(out_dir, config, result)
