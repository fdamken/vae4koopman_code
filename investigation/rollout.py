from typing import Optional, Tuple

import numpy as np

from investigation.observations import compute_observations
from investigation.util import ExperimentConfig, ExperimentResult



def compute_rollout(config: ExperimentConfig, result: ExperimentResult, initial_value: Optional[np.ndarray] = None, T: Optional[int] = None) \
        -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    latent_rollout, latent_cov, latent_rollout_without_control = _compute_latents(config, result, config.T if T is None else T, initial_value)
    obs_rollout, obs_cov = compute_observations(config, result, latent_rollout, latent_cov)
    if latent_rollout_without_control is None:
        obs_rollout_without_control, obs_cov_without_control = None, None
    else:
        obs_rollout_without_control, obs_cov_without_control = compute_observations(config, result, latent_rollout_without_control, latent_cov)
    return (latent_rollout, latent_cov), (obs_rollout, obs_cov), (latent_rollout_without_control, obs_rollout_without_control, obs_cov_without_control)



def _compute_latents(config: ExperimentConfig, result: ExperimentResult, T: int, initial_value: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    Q = np.diag(result.Q)
    rollout = np.zeros((T, config.latent_dim))
    covariances = np.zeros((T, config.latent_dim, config.latent_dim))
    if result.B is None:
        rollout_with_control = None
    else:
        rollout_with_control = np.zeros((T, config.latent_dim))
        rollout_with_control[0, :] = result.m0 if initial_value is None else initial_value
    rollout[0, :] = result.m0 if initial_value is None else initial_value
    covariances[0, :, :] = np.diag(result.V0) if initial_value is None else np.zeros(result.V0.shape)
    for t in range(1, T):
        if result.B is not None:
            rollout_with_control[t, :] = result.A @ rollout_with_control[t - 1, :] + result.B @ result.control_inputs[0, t - 1, :]
        rollout[t, :] = result.A @ rollout[t - 1, :]
        covariances[t, :, :] = result.A @ covariances[t - 1, :, :] @ result.A.T + Q
    covariances = np.asarray([np.diag(x) for x in covariances])
    if rollout_with_control is None:
        return rollout, covariances, None
    return rollout_with_control, covariances, rollout
