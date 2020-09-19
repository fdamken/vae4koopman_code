from typing import Tuple

import numpy as np

from investigation.util import ExperimentConfig, ExperimentResult
from src import cubature
from src.util import outer_batch



def compute_observations(config: ExperimentConfig, result: ExperimentResult, latent_trajectory: np.ndarray, latent_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.diag(result.R)
    latent_covariances_matrices = np.asarray([np.diag(x) for x in latent_covariances])
    observations = cubature.spherical_radial(config.latent_dim, lambda x: result.g_numpy(x), latent_trajectory, latent_covariances_matrices)[0]
    correlations = cubature.spherical_radial(config.latent_dim, lambda x: outer_batch(result.g_numpy(x)), latent_trajectory, latent_covariances_matrices)[0]
    covariances = correlations - outer_batch(observations) + R
    covariances = np.asarray([np.diag(cov) for cov in covariances])
    return observations, covariances
