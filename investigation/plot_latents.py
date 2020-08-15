import numpy as np

from investigation.plot_util import figsize, SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentResult



def plot_latents(out_dir: str, config: ExperimentConfig, result: ExperimentResult):
    domain = np.arange(config.T_train) * config.h

    with SubplotsAndSave(out_dir, 'latents', config.N, 1, sharex = 'all', figsize = figsize(config.N, 1), squeeze = False) as (fig, axs):
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
            ax.legend(loc = 'lower right')
