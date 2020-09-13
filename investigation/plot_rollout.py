from typing import List, Optional

import numpy as np

from investigation.plot_util import figsize, SubplotsAndSave, tuda
from investigation.rollout import compute_rollout
from investigation.util import ExperimentConfig, ExperimentResult



def plot_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, plot_latents: bool, plot_observations: bool):
    if not plot_latents and not plot_observations:
        return

    (latent_rollout, latent_cov), (obs_rollout, obs_cov), without_control = compute_rollout(config, result)
    latent_rollouts = [latent_rollout] * config.N
    latent_covs = [latent_cov] * config.N
    obs_rollouts = [obs_rollout] * config.N
    obs_covs = [obs_cov] * config.N
    latent_rollouts_without_control = [None if without_control is None else without_control[0]] * config.N
    obs_rollouts_without_control = [None if without_control is None else without_control[1]] * config.N

    if plot_latents:
        _plot_latent_rollout(out_dir, config, result, latent_rollouts, latent_covs, latent_rollouts_without_control)
    if plot_observations:
        _plot_observations_rollout(out_dir, config, result, obs_rollouts, obs_covs, obs_rollouts_without_control)



def _plot_latent_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, latent_rollout: List[np.ndarray], latent_covariances: List[np.ndarray],
                         latent_rollout_without_control: List[Optional[np.ndarray]]):
    domain = np.arange(config.T) * config.h
    domain_train = domain[:config.T_train]
    domain_test = domain[config.T_train:]

    with SubplotsAndSave(out_dir, 'rollout-latents', config.latent_dim, config.N,
                         sharex = 'col',
                         sharey = 'row',
                         figsize = figsize(config.latent_dim, config.N),
                         squeeze = False) as (fig, axss):
        for dim, axs in enumerate(axss):
            for n, (ax, latent_trajectory, latent_covariance, latent_trajectory_without_control) in enumerate(
                    zip(axs, latent_rollout, latent_covariances, latent_rollout_without_control)):
                latent_trajectory_train = latent_trajectory[:config.T_train, dim]
                latent_trajectory_test = latent_trajectory[config.T_train:, dim]

                confidence = 2 * np.sqrt(latent_covariance[:, dim])
                upper = latent_trajectory[:, dim] + confidence
                lower = latent_trajectory[:, dim] - confidence

                if latent_rollout_without_control is not None:
                    latent_trajectory_without_control_train = latent_trajectory_without_control[:config.T_train, dim]
                    latent_trajectory_without_control_test = latent_trajectory_without_control[config.T_train:, dim]

                    ax.plot(domain_train, latent_trajectory_without_control_train, color = tuda('pink'), label = 'Rollout w/o Control')
                    ax.plot(domain_test, latent_trajectory_without_control_test, color = tuda('pink'), ls = 'dashed', label = 'Rollout w/o Control (Prediction)')

                ax.plot(domain_train, result.estimations_latents[n, dim, :], color = tuda('orange'), ls = 'dashdot', label = 'Smoothed')
                ax.plot(domain_train, latent_trajectory_train, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_test, latent_trajectory_test, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                ax.scatter(domain[0], result.m0[dim], marker = '*', color = tuda('green'), label = 'Learned Initial Value')
                if dim == 0:
                    ax.set_title('Sequence %d' % (n + 1))
                if dim == config.latent_dim - 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel('Dim. %d' % (dim + 1))
                ax.legend(loc = 'lower left')



def _plot_observations_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, observation_trajectories: List[np.ndarray],
                               observation_covariances: List[np.ndarray], observation_trajectories_without_control: List[Optional[np.ndarray]]):
    domain = np.arange(config.T) * config.h
    domain_train = domain[:config.T_train]
    domain_test = domain[config.T_train:]

    learned_initial_observation = result.g_numpy(result.m0)
    observation_trajectories_smoothed = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape(
            (config.N, config.T_train, config.observation_dim))

    plot_noisy_data = not np.allclose(result.observations_noisy, result.observations)

    with SubplotsAndSave(out_dir, 'rollout-observations', config.observation_dim, config.N,
                         sharex = 'col',
                         sharey = 'row',
                         figsize = figsize(config.observation_dim, config.N),
                         squeeze = False) as (fig, axss):
        for dim, (axs, dim_name) in enumerate(zip(axss, config.observation_dim_names)):
            for n, (ax, observation_trajectory, observation_covariance, observation_trajectory_without_control) in enumerate(
                    zip(axs, observation_trajectories, observation_covariances, observation_trajectories_without_control)):
                observation_trajectory_train = observation_trajectory[:config.T_train, dim]
                observation_trajectory_test = observation_trajectory[config.T_train:, dim]

                confidence = 2 * np.sqrt(observation_covariance[:, dim])
                upper = observation_trajectory[:, dim] + confidence
                lower = observation_trajectory[:, dim] - confidence

                if observation_trajectory_without_control is not None:
                    observation_trajectory_without_control_train = observation_trajectory_without_control[:config.T_train, dim]
                    observation_trajectory_without_control_test = observation_trajectory_without_control[config.T_train:, dim]

                    ax.plot(domain_train, observation_trajectory_without_control_train, color = tuda('pink'), label = 'Rollout w/o Control')
                    ax.plot(domain_test, observation_trajectory_without_control_test, color = tuda('pink'), ls = 'dashed', label = 'Rollout w/o Control (Prediction)')

                ax.scatter(domain, result.observations[n, :, dim], s = 1, color = tuda('black'), label = 'Truth')
                if plot_noisy_data:
                    ax.scatter(domain, result.observations_noisy[n, :, dim], s = 1, color = tuda('black'), alpha = 0.2, label = 'Truth (Noisy)')
                ax.plot(domain_train, observation_trajectories_smoothed[n, :, dim], color = tuda('orange'), ls = 'dashdot', label = 'Smoothed')
                ax.plot(domain_train, observation_trajectory_train, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_test, observation_trajectory_test, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                ax.axvline(domain_train[-1], color = tuda('red'), ls = 'dotted', label = 'Prediction Boundary')
                ax.scatter(domain[0], learned_initial_observation[dim], marker = '*', color = tuda('green'), label = 'Learned Initial Value')
                if dim == 0:
                    ax.set_title('Sequence %d' % (n + 1))
                if dim == config.observation_dim - 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel(dim_name)
                ax.legend(loc = 'lower left')
