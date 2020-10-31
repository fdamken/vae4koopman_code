from typing import List, Optional, Tuple

import numpy as np

from investigation.observations import compute_observations
from investigation.util import ExperimentConfig, ExperimentResult


def compute_rollout(config: ExperimentConfig, result: ExperimentResult, N: int, initial_value: Optional[np.ndarray] = None, T: Optional[int] = None) \
        -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    latent_rollout_without_control = None
    latent_cov = None
    latent_rollouts = []
    latent_covs = []
    obs_rollouts = []
    obs_covs = []
    for n in range(N):
        latent_rollout, latent_cov, latent_rollout_without_control = _compute_latents(config, result, config.T if T is None else T, n, initial_value)
        obs_rollout, obs_cov = compute_observations(config, result, latent_rollout, latent_cov)
        latent_rollouts.append(latent_rollout)
        latent_covs.append(latent_cov)
        obs_rollouts.append(obs_rollout)
        obs_covs.append(obs_cov)
    if latent_rollout_without_control is None:
        obs_rollout_without_control, obs_cov_without_control = None, None
    else:
        obs_rollout_without_control, obs_cov_without_control = compute_observations(config, result, latent_rollout_without_control, latent_cov)
    return (latent_rollouts, latent_covs), (obs_rollouts, obs_covs), (latent_rollout_without_control, obs_rollout_without_control, obs_cov_without_control)


def _compute_latents(config: ExperimentConfig, result: ExperimentResult, T: int, n: int, initial_value: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rollout = np.zeros((T, config.latent_dim))
    covariances = np.zeros((T, config.latent_dim, config.latent_dim))
    if result.B is None:
        rollout_with_control = None
    else:
        rollout_with_control = np.zeros((T, config.latent_dim))
        rollout_with_control[0, :] = result.estimations_latents[n, :, 0] if initial_value is None else initial_value
    rollout[0, :] = result.m0 if initial_value is None else initial_value
    covariances[0, :, :] = result.V0 if initial_value is None else np.zeros(result.V0.shape)
    for t in range(1, T):
        if result.B is not None:
            rollout_with_control[t, :] = result.A @ rollout_with_control[t - 1, :] + result.B @ result.control_inputs[n, t - 1, :]
        if result.B is None or result.neutral_control_input is None:
            rollout[t, :] = result.A @ rollout[t - 1, :]
        else:
            rollout[t, :] = result.A @ rollout[t - 1, :] + result.B @ result.neutral_control_input
        covariances[t, :, :] = result.A @ covariances[t - 1, :, :] @ result.A.T + result.Q
    if rollout_with_control is None:
        return rollout, covariances, None
    return rollout_with_control, covariances, rollout
