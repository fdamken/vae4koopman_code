from typing import List, Union

import numpy as np

from investigation.plot_util import SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentResult, load_run


INIT = 'sample'
TRAJECTORY_LENGTHS = [50, 100, 200]



def generate_latent_trajectory(config: ExperimentConfig, result: ExperimentResult, T: int, init: Union[str, np.ndarray]) -> np.ndarray:
    trajectory = np.zeros((T, config.latent_dim))
    if type(init) == str:
        if init == 'sample':
            trajectory[0, :] = np.random.multivariate_normal(result.m0, result.V0)
        elif init == 'mean':
            trajectory[0, :] = result.m0
        else:
            raise Exception('Unknown initialization method <%s>!' % init)
    elif type(init) == np.ndarray:
        trajectory[0, :] = init
    else:
        raise Exception('Unknown initialization type <%s>!' % str(type(init)))
    for t in range(1, T):
        trajectory[t, :] = result.A @ trajectory[t - 1, :]
    return trajectory



def generate_observation_trajectory(result: ExperimentResult, latent_trajectory: np.ndarray) -> np.ndarray:
    return result.g_numpy(latent_trajectory)



def plot_latents(config: ExperimentConfig, out_dir: str, trajectories: List[np.ndarray]):
    N = len(trajectories)

    with SubplotsAndSave(out_dir, 'generated-latents', 1, N, figsize = (2 + 5 * N, 5), squeeze = False) as (fig, axs):
        for sequence, (ax, trajectory) in enumerate(zip(axs.flatten(), trajectories)):
            T, _ = trajectory.shape
            domain = np.arange(T) * config.h
            for dim in range(config.latent_dim):
                ax.plot(domain, trajectory[:, dim], label = 'Dim. %d' % (dim + 1))
            ax.set_title('Sequence %d, %d Time Steps with Size %.2f' % (sequence + 1, T, config.h))
            if sequence == config.N - 1:
                ax.set_xlabel('Time Steps')
            ax.set_ylabel('Latents')
            ax.legend()
        fig.tight_layout()



def plot_observations(config: ExperimentConfig, out_dir: str, trajectories: List[np.ndarray]):
    N = len(trajectories)

    with SubplotsAndSave(out_dir, 'generated-observations', N, config.observation_dim,
                         sharey = 'row',
                         figsize = (2 + 5 * config.observation_dim, 1 + 4 * N),
                         squeeze = False) as (fig, axss):
        for sequence, (axs, trajectory) in enumerate(zip(axss, trajectories)):
            T, _ = trajectory.shape
            domain = np.arange(T) * config.h
            for dim, ax in enumerate(axs):
                ax.plot(domain, trajectory[:, dim], label = 'Generated')
                ax.set_title('Sequence %d, %d Time Steps with Size %.2f; Dim. %d' % (sequence + 1, T, config.h, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observation')
                ax.legend()
        fig.tight_layout()



def generate_trajectories(out_dir, config: ExperimentConfig, result: ExperimentResult):
    latent_trajectories = []
    observation_trajectories = []
    for T in TRAJECTORY_LENGTHS:
        latent_trajectory = generate_latent_trajectory(config, result, T, INIT)
        observation_trajectory = generate_observation_trajectory(result, latent_trajectory)
        latent_trajectories.append(latent_trajectory)
        observation_trajectories.append(observation_trajectory)

    plot_latents(config, out_dir, latent_trajectories)
    plot_observations(config, out_dir, observation_trajectories)



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, _ = load_run('tmp_results/248', 'checkpoint_00020', None)

    generate_trajectories(out_dir, config, result)
