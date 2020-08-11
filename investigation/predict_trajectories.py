from typing import List

import numpy as np

from investigation.generate_trajectories import generate_latent_trajectory, generate_observation_trajectory
from investigation.plot_util import SubplotsAndSave
from investigation.util import ExperimentConfig, load_run


PREDICTION_STEPS = 25



def plot_trajectories(config: ExperimentConfig, out_dir: str, out_file_name: str, trajectory_type: str, given_trajectories: List[np.ndarray],
                      predicted_trajectories: List[np.ndarray]):
    N = len(given_trajectories)
    _, dims = given_trajectories[0].shape

    with SubplotsAndSave(out_dir, out_file_name, N, dims, sharex = 'row', sharey = 'row', figsize = (2 + 5 * dims, 1 + 4 * N), squeeze = False) as (fig, axss):
        for n, (axs, given_trajectory, predicted_trajectory) in enumerate(zip(axss, given_trajectories, predicted_trajectories)):
            T_given, _ = given_trajectory.shape
            T_predicted, _ = predicted_trajectory.shape
            domain_given = np.arange(T_given) * config.h
            domain_predicted = domain_given[-1] + np.arange(T_predicted) * config.h
            for dim, ax in enumerate(axs):
                ax.plot(domain_given, given_trajectory[:, dim], label = 'Given')
                ax.plot(domain_predicted, predicted_trajectory[:, dim], label = 'Predicted')
                ax.set_title('Sequence %d, %d+%d Time Steps with Size %.2f; Dim. %d' % (n + 1, T_given, T_predicted, config.h, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(trajectory_type)
                ax.legend()
        fig.tight_layout()



def plot_latents(config: ExperimentConfig, out_dir: str, given_trajectories: List[np.ndarray], predicted_trajectories: List[np.ndarray]):
    N = len(given_trajectories)

    with SubplotsAndSave(out_dir, 'predicted-latents', 1, N, figsize = (2 + 5 * N, 5), squeeze = False) as (fig, axs):
        for sequence, (ax, given_trajectory, predicted_trajectory) in enumerate(zip(axs.flatten(), given_trajectories, predicted_trajectories)):
            T_given, _ = given_trajectory.shape
            T_predicted, _ = predicted_trajectory.shape
            domain_given = np.arange(T_given) * config.h
            domain_predicted = domain_given[-1] + np.arange(T_predicted) * config.h
            dim_colors = []
            for dim in range(config.latent_dim):
                line = ax.plot(domain_given, given_trajectory[:, dim], label = 'Dim. %d; Given' % (dim + 1))
                dim_colors.append(line[0].get_color())
            for dim, color in enumerate(dim_colors):
                ax.plot(domain_predicted, predicted_trajectory[:, dim], ls = ':', color = color, label = 'Dim. %d; Predicted' % (dim + 1))
            ax.set_title('Sequence %d, %d+%d Time Steps with Size %.2f' % (n + 1, T_given, T_predicted - 1, config.h))
            if sequence == config.N - 1:
                ax.set_xlabel('Time Steps')
            ax.set_ylabel('Latents')
            ax.legend()
        fig.tight_layout()



def plot_observations(config: ExperimentConfig, out_dir: str, given_trajectories: List[np.ndarray], predicted_trajectories: List[np.ndarray]):
    N = len(given_trajectories)

    with SubplotsAndSave(out_dir, 'predicted-observations', N, config.observation_dim,
                         sharey = 'row',
                         figsize = (2 + 5 * config.observation_dim, 1 + 4 * N),
                         squeeze = False) as (fig, axss):
        for sequence, (axs, given_trajectory, predicted_trajectory) in enumerate(zip(axss, given_trajectories, predicted_trajectories)):
            T_given, _ = given_trajectory.shape
            T_predicted, _ = predicted_trajectory.shape
            domain_given = np.arange(T_given) * config.h
            domain_predicted = domain_given[-1] + np.arange(T_predicted) * config.h
            for dim, ax in enumerate(axs):
                line = ax.plot(domain_given, given_trajectory[:, dim], label = 'Given')
                ax.plot(domain_predicted, predicted_trajectory[:, dim], ls = ':', color = line[0].get_color(), label = 'Predicted')
                ax.set_title('Sequence %d, %d+%d Time Steps with Size %.2f; Dim. %d' % (sequence + 1, T_given, T_predicted - 1, config.h, dim + 1))
                if sequence == config.N - 1:
                    ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observation')
                ax.legend()
        fig.tight_layout()



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, _ = load_run('tmp_results/248', 'checkpoint_00020', None)
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

    # plot_trajectories(config, out_dir, 'predicted-latents', 'Latents', given_latent_trajectories, predicted_latent_trajectories)
    # plot_trajectories(config, out_dir, 'predicted-observations', 'Observations', given_observation_trajectories, predicted_observation_trajectories)

    plot_latents(config, out_dir, given_latent_trajectories, predicted_latent_trajectories)
    plot_observations(config, out_dir, given_observation_trajectories, predicted_observation_trajectories)
