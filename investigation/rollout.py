from typing import Optional, Tuple

import numpy as np

from investigation.util import ExperimentConfig, ExperimentResult
from src import cubature
from src.util import outer_batch



def compute_rollout(config: ExperimentConfig, result: ExperimentResult, initial_value: Optional[np.ndarray] = None, T: Optional[int] = None) \
        -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    latent_rollout, latent_cov, latent_rollout_without_control = _compute_latents(config, result, config.T if T is None else T, initial_value)
    obs_rollout, obs_cov = _compute_observations(config, result, latent_rollout, latent_cov)
    if latent_rollout_without_control is None:
        obs_rollout_without_control, obs_cov_without_control = None, None
    else:
        obs_rollout_without_control, obs_cov_without_control = _compute_observations(config, result, latent_rollout_without_control, latent_cov)
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



def _compute_observations(config: ExperimentConfig, result: ExperimentResult, latent_trajectory: np.ndarray, latent_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.diag(result.R)
    latent_covariances_matrices = np.asarray([np.diag(x) for x in latent_covariances])
    observations = cubature.spherical_radial(config.latent_dim, lambda x: result.g_numpy(x), latent_trajectory, latent_covariances_matrices)[0]
    correlations = cubature.spherical_radial(config.latent_dim, lambda x: outer_batch(result.g_numpy(x)), latent_trajectory, latent_covariances_matrices)[0]
    covariances = correlations - outer_batch(observations) + R
    covariances = np.asarray([np.diag(cov) for cov in covariances])
    # "Remove" values close to zero to avoid invalid square-roots with negative numbers due to numerical instabilities. A value
    # slightly more than zero (e.g. 1e-8) does not remove the covariance at all resulting in a weird plot, but removes just enough
    # to visualize that the model is extremely confident while not looking weird.
    covariances[np.isclose(covariances, 0.0, rtol = 1e-2, atol = 1e-5)] = 1e-8
    return observations, covariances
