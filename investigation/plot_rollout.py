from typing import List

import numpy as np

from investigation.plot_util import figsize, SubplotsAndSave, tuda
from investigation.rollout import compute_rollout
from investigation.util import ExperimentConfig, ExperimentResult



def plot_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, plot_latents: bool, plot_observations: bool):
    if not plot_latents and not plot_observations:
        return

    latent_rollouts, latent_covs, obs_rollouts, obs_covs = [], [], [], []
    for n in range(config.N):
        (latent_rollout, latent_cov), (obs_rollout, obs_cov) = compute_rollout(config, result, initial_value = result.observations[n, 0, :])
        latent_rollouts.append(latent_rollout)
        latent_covs.append(latent_cov)
        obs_rollouts.append(obs_rollout)
        obs_covs.append(obs_cov)

    if plot_latents:
        _plot_latent_rollout(out_dir, config, result, latent_rollouts, latent_covs)
    if plot_observations:
        _plot_observations_rollout(out_dir, config, result, obs_rollouts, obs_covs)



def _plot_latent_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, latent_rollout: List[np.ndarray], latent_covariances: List[np.ndarray]):
    domain = np.arange(config.T) * config.h
    domain_train = domain[:config.T_train]
    domain_test = domain[config.T_train:]

    with SubplotsAndSave(out_dir, 'latents-rollout', config.latent_dim, config.N,
                         sharex = 'col',
                         sharey = 'row',
                         figsize = figsize(config.latent_dim, config.N),
                         squeeze = False) as (fig, axss):
        for dim, axs in enumerate(axss):
            for n, (ax, latent_trajectory, latent_covariance) in enumerate(zip(axs, latent_rollout, latent_covariances)):
                latent_trajectory_train = latent_trajectory[:config.T_train, :, dim]
                latent_trajectory_test = latent_trajectory[config.T_train:, :, dim]

                confidence = 2 * np.sqrt(latent_covariance[:, dim])
                upper = latent_trajectory[:, dim] + confidence
                lower = latent_trajectory[:, dim] - confidence

                # TODO: Maybe m0 has to be unsqueezed.
                ax.scatter(domain[0], result.m0[dim], '*', color = tuda('purple'), label = 'Learned Initial Value')
                ax.plot(domain_train, result.estimations_latents[n, dim, :], color = tuda('orange'), ls = 'dashdot', label = 'Smoothed')
                ax.plot(domain_train, latent_trajectory_train, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_test, latent_trajectory_test, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                if dim == 0:
                    ax.set_title('Sequence %d' % (n + 1))
                if dim == config.latent_dim - 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel('Dim. %d' % (dim + 1))
                ax.legend(loc = 'lower right')



def _plot_observations_rollout(out_dir: str, config: ExperimentConfig, result: ExperimentResult, observation_trajectories: List[np.ndarray],
                               observation_covariances: List[np.ndarray]):
    domain = np.arange(config.T) * config.h
    domain_train = domain[:config.T_train]
    domain_test = domain[config.T_train:]

    # TODO: Maybe m0 has to be unsqueezed.
    learned_initial_observation = result.g_numpy(result.m0)[0, :]
    observation_trajectories_smoothed = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape(
            (config.N, config.T, config.observation_dim))

    with SubplotsAndSave(out_dir, 'observations-rollout', config.observation_dim, config.N,
                         sharex = 'col',
                         sharey = 'row',
                         figsize = figsize(config.observation_dim, config.N),
                         squeeze = False) as (fig, axss):
        for dim, (axs, dim_name) in enumerate(zip(axss, config.observation_dim_names)):
            for n, (ax, observation_trajectory, observation_covariance) in enumerate(zip(axs, observation_trajectories, observation_covariances)):
                observation_trajectory_train = observation_trajectory[:config.T_train, :, dim]
                observation_trajectory_test = observation_trajectory[config.T_train:, :, dim]

                confidence = 2 * np.sqrt(observation_covariance[:, dim])
                upper = observation_trajectory + confidence
                lower = observation_trajectory - confidence

                ax.scatter(domain[0], learned_initial_observation[dim], '*', color = tuda('purple'), label = 'Learned Initial Value')
                ax.scatter(domain, result.observations[n, :, dim], s = 1, color = tuda('black'), label = 'Truth')
                ax.plot(domain_train, observation_trajectories_smoothed[n, :, dim], color = tuda('orange'), ls = 'dashdot', label = 'Smoothed')
                ax.plot(domain_train, observation_trajectory_train, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_test, observation_trajectory_test, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                ax.axvline(domain_train[-1], color = tuda('red'), ls = 'dotted', label = 'Prediction Boundary')
                if dim == 0:
                    ax.set_title('Sequence %d' % (n + 1))
                if dim == 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel(dim_name)
                ax.legend(loc = 'lower right')
