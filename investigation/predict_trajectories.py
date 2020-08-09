from typing import List

import matplotlib.pyplot as plt
import numpy as np

from investigation.generate_trajectories import generate_latent_trajectory, generate_observation_trajectory
from investigation.util import load_run


PREDICTION_STEPS = 5000



def plot_trajectories(ex_title: str, trajectory_type: str, h: float, given_trajectories: List[np.ndarray], predicted_trajectories: List[np.ndarray]):
    N = len(given_trajectories)
    _, dims = given_trajectories[0].shape

    fig, axss = plt.subplots(N, dims, sharex = 'row', sharey = 'row', figsize = (2 + 5 * dims, 5 * N), squeeze = False)
    for n, (axs, given_trajectory, predicted_trajectory) in enumerate(zip(axss, given_trajectories, predicted_trajectories)):
        T_given, _ = given_trajectory.shape
        T_predicted, _ = predicted_trajectory.shape
        domain_given = np.arange(T_given) * h
        domain_predicted = domain_given[-1] + np.arange(T_predicted) * h
        for dim, ax in enumerate(axs):
            ax.plot(domain_given, given_trajectory[:, dim], label = 'Given')
            ax.plot(domain_predicted, predicted_trajectory[:, dim], label = 'Predicted')
            ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(trajectory_type)
            ax.legend()
    fig.suptitle('Given/Predicted %s (%s)' % (trajectory_type, ex_title))
    fig.show()



if __name__ == '__main__':
    config, result = load_run('tmp_results/138', 'run')
    observations = result.observations
    latents = result.estimations_latents

    N, T, _ = latents.shape
    given_latent_trajectories = []
    predicted_latent_trajectories = []
    given_observation_trajectories = []
    predicted_observation_trajectories = []
    for n in range(N):
        given_latent_trajectories.append(latents[n, :, :].T)
        predicted_latent_trajectories.append(generate_latent_trajectory(config, result, PREDICTION_STEPS + 1, given_latent_trajectories[-1][-1, :]))
        given_observation_trajectories.append(result.observations[n, :, :])
        predicted_observation_trajectories.append(generate_observation_trajectory(result, predicted_latent_trajectories[-1]))

    plot_trajectories(config.title, 'Latents', config.h, given_latent_trajectories, predicted_latent_trajectories)
    plot_trajectories(config.title, 'Observations', config.h, given_observation_trajectories, predicted_observation_trajectories)
