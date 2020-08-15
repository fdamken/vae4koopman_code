import numpy as np
import scipy.integrate as sci

from investigation.plot_rollout import compute_observations, compute_rollout, title_for_dim
from investigation.plot_util import SubplotsAndSave, tuda
from investigation.util import load_run


TEST_TIME_STEPS = 500



def compute_trajectory(t_start: float, t_final: float, T: int, h: float, param_damping: float, initial_value: np.ndarray) -> np.ndarray:
    if not np.isclose(T * h, t_final):
        raise Exception('T, h and t_final are inconsistent! The result of T * h should equal t_final.')

    ode = lambda x1, x2: np.asarray([x2,
                                     np.sin(x1) - param_damping * x2])

    return sci.solve_ivp(lambda t, x: ode(*x), (t_start, t_final), initial_value, t_eval = np.arange(t_start, t_final, h), method = 'Radau').y.T



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'

    result_dir = 'latent-dims-experiment/log-100-100/results'
    subdirs = range(44, 55 + 1)
    configs, results, metricss = zip(*[load_run('%s/%d' % (result_dir, subdir), 'run', 'metrics') for subdir in subdirs])
    N = configs[0].N
    T = configs[0].T
    h = configs[0].h
    observations = results[0].observations
    observation_dim = configs[0].observation_dim
    param_damping = configs[0].param_damping
    initial_value_mean = configs[0].initial_value_mean

    assert N == 1, 'Experiments a different number of sequences than one!'
    assert observation_dim == 2, 'Observation dim is different than two!'
    assert all([config.N == N for config in configs]), 'Some experiments have trained on different number of sequences!'
    assert all([config.T == T for config in configs]), 'Some experiments have trained on different sequence lengths!'
    assert all([config.h == h for config in configs]), 'Some experiments have trained on sequences with different time step widths!'
    assert all([np.allclose(result.observations, observations) for result in results]), 'Some experiments have trained on different trajectories!'
    assert all([config.param_damping == param_damping for config in configs]), 'Some experiments have trained with different damping parameters!'
    assert all([np.allclose(config.initial_value_mean, initial_value_mean) for config in configs]), 'Some experiments have trained with different initial values!'

    T_with_test = T + TEST_TIME_STEPS
    t_final = T_with_test * h
    trajectory = np.concatenate([observations[0, :, :], compute_trajectory(T * h, t_final + h, T_with_test + 1, h, param_damping, observations[0, -1, :].flatten())[1:, :]],
                                axis = 0)
    assert np.allclose(results[0].observations, trajectory[:T, :])
    rollouts, rollout_latent_covariances = zip(*[compute_rollout(config, result, T_with_test) for config, result in zip(configs, results)])
    observation_rollouts, observation_covariances = zip(
            *[compute_observations(config, result, rollout, cov) for config, result, rollout, cov in zip(configs, results, rollouts, rollout_latent_covariances)])
    observations_smoothed = [result.g_numpy(result.estimations_latents[0, :, :].T) for result in results]
    observations_smoothed = [
            result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape((config.N, config.T, config.observation_dim))[0, :, :] for
            config, result in zip(configs, results)]

    # Compute mean squared errors per run.
    mse_data = []
    for config, observation_rollout in zip(configs, observation_rollouts):
        mse_per_dim = np.linalg.norm(trajectory - observation_rollout, axis = 0) ** 2 / trajectory.shape[0]
        mse = np.sum(mse_per_dim) / mse_per_dim.shape[0]
        mse_data.append((config.latent_dim, *mse_per_dim, mse))
    mse_data = np.asarray(mse_data)

    # Plot rollout.
    domain = np.arange(T_with_test) * h
    domain_before_prediction = domain[:T]
    domain_after_prediction = domain[T:]
    assert len(domain_before_prediction) == T
    assert len(domain_after_prediction) == TEST_TIME_STEPS
    latent_dims_to_keep = [5, 10, 60]
    with SubplotsAndSave(out_dir, 'latent-dimensions-experiment_rollout', observation_dim, len(latent_dims_to_keep),
                         sharex = 'col',
                         sharey = 'row',
                         figsize = (1 + 5 * len(latent_dims_to_keep), 2 + 4 * observation_dim),
                         squeeze = False) as (fig, axss):
        tmp = [(c, r, o_r, o_c, o_s) for c, r, o_r, o_c, o_s in zip(configs, results, observation_rollouts, observation_covariances, observations_smoothed) if
               c.latent_dim in latent_dims_to_keep]
        for dim, axs in enumerate(axss):
            for n, (ax, (config, result, observation_rollout, observation_covariance, observation_smoothed)) in enumerate(zip(axs, tmp)):
                mean = observation_rollout[:, dim]
                mean_before_prediction = mean[:T]
                mean_after_prediction = mean[T:]
                assert len(mean_before_prediction) == T
                assert len(mean_after_prediction) == TEST_TIME_STEPS
                confidence = 2 * np.sqrt(observation_covariance[:, dim])
                upper = mean + confidence
                lower = mean - confidence

                ax.scatter(domain, trajectory[:, dim], s = 1, color = tuda('black'), label = 'Truth')
                ax.plot(domain_before_prediction, observation_smoothed[:, dim], color = tuda('orange'), ls = 'dashdot', label = 'Smoothed')
                ax.plot(domain_before_prediction, mean_before_prediction, color = tuda('blue'), label = 'Rollout')
                ax.plot(domain_after_prediction, mean_after_prediction, color = tuda('blue'), ls = 'dashed', label = 'Rollout (Prediction)')
                ax.fill_between(domain, upper, lower, where = upper > lower, color = tuda('blue'), alpha = 0.2, label = 'Rollout Confidence')
                ax.axvline(domain_before_prediction[-1], color = tuda('red'), ls = 'dotted', label = 'Prediction Boundary')
                if dim == 0:
                    ax.set_title('Latent Dimensionality: %d' % config.latent_dim)
                if dim == 1:
                    ax.set_xlabel('Time Steps')
                if n == 0:
                    ax.set_ylabel(title_for_dim(dim))
                ax.legend(loc = 'lower right')
        fig.tight_layout()

    # Plot mean squared error.
    with SubplotsAndSave(out_dir, 'latent-dimensions-experiment_mse') as (fig, ax):
        domain = mse_data[:, 0]
        mse_dim_1 = mse_data[:, 1]
        mse_dim_2 = mse_data[:, 2]
        mse = mse_data[:, 3]

        ax.plot(domain, mse, '-o', color = tuda('blue'), label = 'Total')
        ax.set_title('Mean Squared Error vs. Latent Dimensionality')
        ax.set_yscale('log')
        ax.set_xlabel('Latent Dimensionality')
        ax.set_ylabel('Mean Squared Error')
        ax.legend()
        fig.tight_layout()
