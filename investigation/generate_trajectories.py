from typing import List, Union

import numpy as np

from investigation.plot_util import SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentResult, load_run


INIT = 'sample'
TRAJECTORY_LENGTHS = [500, 1000, 2000, 5000]



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



def plot_trajectories(config: ExperimentConfig, out_dir: str, out_file_name: str, trajectory_type: str, trajectories: List[np.ndarray]):
    N = len(trajectories)
    _, dims = trajectories[0].shape

    with SubplotsAndSave(out_dir, out_file_name, N, dims, sharex = 'row', sharey = 'row', figsize = (2 + 5 * dims, 1 + 4 * N), squeeze = False) as (fig, axss):
        for n, (axs, trajectory) in enumerate(zip(axss, trajectories)):
            T, _ = trajectory.shape
            domain = np.arange(T) * config.h
            for dim, ax in enumerate(axs):
                ax.plot(domain, trajectory[:, dim], label = 'Generated')
                ax.set_title('Trajectory %d, Dim. %d' % (n + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(trajectory_type)
                ax.legend()
        fig.tight_layout()



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result = load_run('tmp_results/138', 'run')

    latent_trajectories = []
    observation_trajectories = []
    for T in TRAJECTORY_LENGTHS:
        latent_trajectory = generate_latent_trajectory(config, result, T, INIT)
        observation_trajectory = generate_observation_trajectory(result, latent_trajectory)
        latent_trajectories.append(latent_trajectory)
        observation_trajectories.append(observation_trajectory)

    plot_trajectories(config, out_dir, 'generated-latents', 'Latents', latent_trajectories)
    plot_trajectories(config, out_dir, 'generated-observations', 'Observations', observation_trajectories)
