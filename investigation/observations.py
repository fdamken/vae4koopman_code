from typing import Tuple

import numpy as np

from investigation.util import ExperimentConfig, ExperimentResult
from src import cubature
from src.util import outer_batch



def compute_observations(config: ExperimentConfig, result: ExperimentResult, latent_trajectory: np.ndarray, latent_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    observations, _, _, cov = cubature.spherical_radial(config.latent_dim, lambda x: result.g_numpy(x), latent_trajectory, latent_covariances)
    correlations = cubature.spherical_radial(config.latent_dim, lambda x: outer_batch(result.g_numpy(x)), latent_trajectory, cov, True)[0]
    covariances = correlations - outer_batch(observations) + result.R
    if result.y_shift is not None and result.y_scale is not None:
        observations = result.y_shift + observations * result.y_scale
    return observations, covariances
