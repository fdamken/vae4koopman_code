import numpy as np

from investigation.plot_util import figsize, SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentResult



def plot_observations(out_dir: str, config: ExperimentConfig, result: ExperimentResult):
    domain = np.arange(config.T) * config.h

    reconstructed_states = result.g_numpy(result.estimations_latents.transpose((0, 2, 1)).reshape(-1, config.latent_dim)).reshape((config.N, config.T, config.observation_dim))
    with SubplotsAndSave(out_dir, 'observations', config.N, config.observation_dim,
                         sharex = 'all',
                         sharey = 'row',
                         figsize = figsize(config.N, config.observation_dim),
                         squeeze = False) as (fig, axss):
        for sequence, axs in enumerate(axss):
            for dim, ax in enumerate(axs):
                ax.plot(domain, result.observations[sequence, :, dim], label = 'Input (without noise)')
                ax.plot(domain, result.observations_noisy[sequence, :, dim], alpha = 0.2, label = 'Input (Noisy)')
                ax.plot(domain, reconstructed_states[sequence, :, dim], ls = '-.', label = 'Reconstructed')
                ax.set_title('Sequence %d, Dim. %d' % (sequence + 1, dim + 1))
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Observation')
                ax.legend(loc = 'lower right')
