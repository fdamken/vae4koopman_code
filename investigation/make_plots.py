import matplotlib.pyplot as plt
import numpy as np

from investigation.util import ExperimentConfig, ExperimentMetrics, ExperimentResult, load_run



def plot_log_likelihood(config: ExperimentConfig, result: ExperimentResult, metrics: ExperimentMetrics, out_dir: str):
    domain = np.arange(result.iterations)
    log_likelihood = metrics.log_likelihood

    fig, ax = plt.subplots()
    ax.plot(domain, log_likelihood, label = 'Log-Likelihood')
    ax.set_title('Log-Likelihood (%s), %d Iterations' % (config.title, result.iterations))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Log-Likelihood')
    ax.legend()
    fig.savefig('%s/log-likelihood.png' % out_dir)



def plot_observations(config: ExperimentConfig, result: ExperimentResult, out_dir: str):
    domain = np.arange(config.T) * config.h

    reconstructed_states = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape((config.N, config.T, config.observation_dim))
    fig, axss = plt.subplots(config.N, config.observation_dim, sharex = 'all', sharey = 'row', figsize = (2 + 5 * config.observation_dim, 1 + 4 * config.N), squeeze = False)
    for sequence, axs in enumerate(axss):
        for dim, ax in enumerate(axs):
            ax.plot(domain, result.observations[sequence, :, dim], label = 'Input (without noise)')
            ax.plot(domain, result.observations_noisy[sequence, :, dim], alpha = 0.2, label = 'Input (Noisy)')
            ax.plot(domain, reconstructed_states[sequence, :, dim], ls = '-.', label = 'Reconstructed')
            ax.set_title('Sequence %d, Dim. %d' % (sequence + 1, dim + 1))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Observation')
            ax.legend()
    fig.suptitle('Input and Reconstructed Observations (%s), %d Iterations' % (config.title, result.iterations))
    fig.tight_layout()
    fig.subplots_adjust(top = 0.9)
    fig.savefig('%s/observations.png' % out_dir)



def plot_latents(config: ExperimentConfig, result: ExperimentResult, out_dir: str):
    domain = np.arange(config.T) * config.h

    fig, axs = plt.subplots(config.N, 1, sharex = 'all', figsize = (8, 1 + 4 * config.N), squeeze = False)
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
    fig.suptitle('Latents (%s), %d Iterations' % (config.title, result.iterations), y = 0.98)
    fig.tight_layout()
    fig.subplots_adjust(top = 0.9)
    fig.savefig('%s/latents.png' % out_dir)



if __name__ == '__main__':
    config, result, metrics = load_run('tmp_results/138', 'run', 'metrics')
    out_dir = 'investigation/tmp_figures'

    plot_log_likelihood(config, result, metrics, out_dir)
    plot_observations(config, result, out_dir)
    plot_latents(config, result, out_dir)
