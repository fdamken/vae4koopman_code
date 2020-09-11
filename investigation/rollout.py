from typing import Optional, Tuple

import numpy as np

from investigation.util import ExperimentConfig, ExperimentResult
from src import cubature
from src.util import outer_batch



def compute_rollout(config: ExperimentConfig, result: ExperimentResult, initial_value: Optional[np.ndarray] = None, T: Optional[int] = None) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    latent_rollout, latent_cov = _compute_latents(config, result, config.T if T is None else T, initial_value)
    obs_rollout, obs_cov = _compute_observations(config, result, latent_rollout, latent_cov)
    return (latent_rollout, latent_cov), (obs_rollout, obs_cov)



def _compute_latents(config: ExperimentConfig, result: ExperimentResult, T: int, initial_value: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.diag(result.Q)
    rollout = np.zeros((T, config.latent_dim))
    covariances = np.zeros((T, config.latent_dim, config.latent_dim))
    rollout[0, :] = result.m0 if initial_value is None else initial_value
    covariances[0, :, :] = np.diag(result.V0) if initial_value is None else np.zeros(result.V0.shape)
    for t in range(1, T):
        if result.B is None:
            rollout[t, :] = result.A @ rollout[t - 1, :]
        else:
            rollout[t, :] = result.A @ rollout[t - 1, :] + result.B @ result.control_inputs[0, t - 1, :]
        covariances[t, :, :] = result.A @ covariances[t - 1, :, :] @ result.A.T + Q
    covariances = np.asarray([np.diag(x) for x in covariances])
    return rollout, covariances



def _compute_observations(config: ExperimentConfig, result: ExperimentResult, latent_trajectory: np.ndarray, latent_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.diag(result.R)
    latent_covariances_matrices = np.asarray([np.diag(x) for x in latent_covariances])
    observations = cubature.spherical_radial(config.latent_dim, lambda x: result.g_numpy(x), latent_trajectory, latent_covariances_matrices)[0]
    correlations = cubature.spherical_radial(config.latent_dim, lambda x: outer_batch(result.g_numpy(x)), latent_trajectory, latent_covariances_matrices)[0]
    covariances = correlations - outer_batch(observations) + R
    covariances = np.asarray([np.diag(cov) for cov in covariances])
    observations_not_expected = result.g_numpy(latent_trajectory)
    return observations_not_expected, covariances
