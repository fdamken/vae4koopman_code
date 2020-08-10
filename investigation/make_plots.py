import numpy as np

from investigation.plot_util import SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentMetrics, ExperimentResult, load_run



def plot_log_likelihood(config: ExperimentConfig, result: ExperimentResult, metrics: ExperimentMetrics, out_dir: str):
    domain = np.arange(result.iterations)
    log_likelihood = metrics.log_likelihood

    with SubplotsAndSave(out_dir, 'log-likelihood', figsize = (7, 5)) as (fig, ax):
        ax.plot(domain, log_likelihood, label = 'Log-Likelihood')
        ax.set_title('Log-Likelihood (%s), %d Iterations' % (config.title, result.iterations))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Log-Likelihood')
        ax.legend()
        fig.tight_layout()



def plot_g_final_log_likelihood(config: ExperimentConfig, result: ExperimentResult, metrics: ExperimentMetrics, out_dir: str):
    domain = np.arange(result.iterations)
    g_final_log_likelihood = metrics.g_final_log_likelihood

    with SubplotsAndSave(out_dir, 'g-final-log-likelihood', figsize = (7, 5)) as (fig, ax):
        ax.plot(domain, g_final_log_likelihood, label = 'G-Final Log-Likelihood')
        ax.set_title('G-Final Log-Likelihood (%s), %d Iterations' % (config.title, result.iterations))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Log-Likelihood')
        ax.legend()
        fig.tight_layout()



def plot_latents(config: ExperimentConfig, result: ExperimentResult, out_dir: str):
    domain = np.arange(config.T) * config.h

    with SubplotsAndSave(out_dir, 'latents', config.N, 1, sharex = 'all', figsize = (7, 1 + 4 * config.N), squeeze = False) as (fig, axs):
        with_label = True
        for sequence, ax in enumerate(axs.flatten()):
            for dim in range(config.latent_dim):
                label = ('Dim. %d' % (dim + 1)) if with_label else None
                ax.plot(domain, result.estimations_latents[sequence, dim, :], label = label)
            with_label = False
            ax.set_title('Sequence %d' % (sequence + 1))
            if sequence == config.N - 1:
                ax.set_xlabel('Time Steps')
            ax.set_ylabel('Latents')
            ax.legend()
        fig.tight_layout()



def plot_observations(config: ExperimentConfig, result: ExperimentResult, out_dir: str):
    domain = np.arange(config.T) * config.h

    reconstructed_states = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape((config.N, config.T, config.observation_dim))
    with SubplotsAndSave(out_dir, 'observations', config.N, config.observation_dim,
                         sharex = 'all',
                         sharey = 'row',
                         figsize = (2 + 5 * config.observation_dim, 1 + 4 * config.N),
                         squeeze = False) as (fig, axss):
        for sequence, axs in enumerate(axss):
            for dim, ax in enumerate(axs):
                ax.plot(domain, result.observations[sequence, :, dim], label = 'Input (without noise)')
                ax.plot(domain, result.observations_noisy[sequence, :, dim], alpha = 0.2, label = 'Input (Noisy)')
                ax.plot(domain, reconstructed_states[sequence, :, dim], ls = '-.', label = 'Reconstructed')
                ax.set_title('Sequence %d, Dim. %d' % (sequence + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observation')
                ax.legend()
        fig.tight_layout()



if __name__ == '__main__':
    out_dir = 'investigation/tmp_figures'
    config, result, metrics = load_run('tmp_results/138', 'run', 'metrics')

    plot_log_likelihood(config, result, metrics, out_dir)
    plot_g_final_log_likelihood(config, result, metrics, out_dir)
    plot_latents(config, result, out_dir)
    plot_observations(config, result, out_dir)
