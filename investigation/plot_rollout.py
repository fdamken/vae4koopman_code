from typing import List, Tuple

import numpy as np
import scipy.integrate as sci

from investigation.plot_util import SubplotsAndSave, tuda
from investigation.util import ExperimentConfig, ExperimentResult, load_run
from src import cubature
from src.util import outer_batch


PREDICTION_STEPS = 500



def compute_trajectory(t_final: float, T: int, h: float, param_damping: float, initial_value: np.ndarray) -> np.ndarray:
    if not np.isclose(T * h, t_final):
        raise Exception('T, h and t_final are inconsistent! The result of T * h should equal t_final.')

    ode = lambda x1, x2: np.asarray([x2,
                                     np.sin(x1) - param_damping * x2])

    return sci.solve_ivp(lambda t, x: ode(*x), (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau').y.T



def compute_rollout(config: ExperimentConfig, result: ExperimentResult, T: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.diag(result.Q)
    rollout = np.zeros((T, config.latent_dim))
    covariances = np.zeros((T, config.latent_dim, config.latent_dim))
    rollout[0, :] = result.m0
    covariances[0, :, :] = np.diag(result.V0)
    for t in range(1, T):
        rollout[t, :] = result.A @ rollout[t - 1, :]
        covariances[t, :, :] = result.A @ covariances[t - 1, :, :] @ result.A.T + Q
    covariances = np.asarray([np.diag(x) for x in covariances])
    return rollout, covariances



def compute_observations(config: ExperimentConfig, result: ExperimentResult, latent_trajectory: np.ndarray, latent_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.diag(result.R)
    latent_covariances_matrices = np.asarray([np.diag(x) for x in latent_covariances])
    observations = cubature.spherical_radial(config.latent_dim, lambda x: result.g_numpy(x), latent_trajectory, latent_covariances_matrices)[0]
    correlations = cubature.spherical_radial(config.latent_dim, lambda x: outer_batch(result.g_numpy(x)), latent_trajectory, latent_covariances_matrices)[0]
    covariances = correlations - outer_batch(observations) + R
    covariances = np.asarray([np.diag(cov) for cov in covariances])
    observations_not_expected = result.g_numpy(latent_trajectory)
    return observations_not_expected, covariances



def title_for_dim(dim: int) -> str:
    if dim == 0:
        return 'Position'
    elif dim == 1:
        return 'Velocity'
    else:
        raise Exception('Invalid observation dimension %d!' % dim)



def plot_latent_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, prediction_steps: int, latent_trajectories: List[np.ndarray],
                        latent_covariances: List[np.ndarray]):
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
        for n, (axs, latent_trajectory, latent_covariance) in enumerate(zip(axss, latent_trajectories, latent_covariances)):
            if prediction_steps > 0:
                latent_trajectory_before_prediction = latent_trajectory[:-prediction_steps, :]
            else:
                latent_trajectory_before_prediction = latent_trajectory
            latent_trajectory_predicted = latent_trajectory[-prediction_steps:, :]
            for dim, ax in enumerate(axs):
                confidence = 2 * np.sqrt(latent_covariance[:, dim])
                upper = latent_trajectory[:, dim] + confidence
                lower = latent_trajectory[:, dim] - confidence

                ax.plot(domain_before_prediction, result.estimations_latents[n, dim, :], color = tuda('orange'), label = 'Filtered/Smoothed')
                ax.plot(domain_before_prediction, latent_trajectory_before_prediction[:, dim], color = tuda('blue'), label = 'Rollout')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                if prediction_steps > 0:
                    ax.plot(domain_predicted, latent_trajectory_predicted[:, dim], color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Latents')
                ax.legend()
        fig.tight_layout()



def plot_observations_rollout(config: ExperimentConfig, result: ExperimentResult, out_dir: str, prediction_steps: int, real_trajectory: np.ndarray,
                              observation_trajectories: List[np.ndarray], observation_covariances: List[np.ndarray]):
    T, _ = observation_trajectories[0].shape
    domain = np.arange(T) * config.h
    if prediction_steps > 0:
        domain_before_prediction = domain[:-prediction_steps]
    else:
        domain_before_prediction = domain
    domain_predicted = domain[-prediction_steps:]

    observation_trajectories_smoothed = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape(
            (config.N, config.T, config.observation_dim))
    with SubplotsAndSave(out_dir, 'rollout-observations', config.observation_dim, config.N,
                         sharex = 'all',
                         figsize = (1 + 5 * config.N, 2 + 4 * config.observation_dim),
                         squeeze = False) as (fig, axss):
        for dim, axs in enumerate(axss):
            for n, (ax, observation_trajectory, observation_trajectory_smoothed, observation_covariance) in enumerate(
                    zip(axs, observation_trajectories, observation_trajectories_smoothed, observation_covariances)):
                mean = observation_trajectory[:, dim]
                confidence = 2 * np.sqrt(observation_covariance[:, dim])
                upper = mean + confidence
                lower = mean - confidence
                if prediction_steps > 0:
                    mean_before_prediction = mean[:-prediction_steps]
                else:
                    mean_before_prediction = mean
                mean_predicted = mean[-prediction_steps:]

                ax.scatter(domain, real_trajectory[:, dim], s = 1, color = tuda('black'), label = 'Truth')
                ax.plot(domain_before_prediction, mean_before_prediction, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_predicted, mean_predicted, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                # ax.plot(domain, observation_trajectory_smoothed[:, dim], ls = 'dashdot', label = 'Smoothed/Filtered')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                if prediction_steps > 0:
                    ax.axvline(domain_before_prediction[-1], color = tuda('red'), ls = 'dotted', label = 'Prediction Boundary')
                ax.set_title('%s' % title_for_dim(dim))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observations')
                ax.legend()
        fig.tight_layout()



def plot_rollout(out_dir, config: ExperimentConfig, result: ExperimentResult):
    T_with_prediction = config.T + PREDICTION_STEPS
    real_trajectory = compute_trajectory(T_with_prediction * config.h, T_with_prediction, config.h, config.param_damping, config.initial_value_mean)
    latent_trajectories = []
    latent_covariances = []
    observation_trajectories = []
    observation_covariances = []
    for _ in range(config.N):
        latent_trajectory, latent_covariance = compute_rollout(config, result, config.T + PREDICTION_STEPS)
        observation_trajectory, observation_covariance = compute_observations(config, result, latent_trajectory, latent_covariance)
        latent_trajectories.append(latent_trajectory)
        latent_covariances.append(latent_covariance)
        observation_trajectories.append(observation_trajectory)
        observation_covariances.append(observation_covariance)

    plot_latent_rollout(config, result, out_dir, PREDICTION_STEPS, latent_trajectories, latent_covariances)
    plot_observations_rollout(config, result, out_dir, PREDICTION_STEPS, real_trajectory, observation_trajectories, observation_covariances)



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, _ = load_run('latent-dims-experiment/log-100-100/results/52', 'run', 'metrics')

    plot_rollout(out_dir, config, result)
