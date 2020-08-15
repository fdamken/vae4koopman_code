import numpy as np

from investigation.plot_util import figsize, SubplotsAndSave
from investigation.util import ExperimentConfig, ExperimentMetrics, ExperimentResult



def plot_log_likelihood(out_dir: str, config: ExperimentConfig, result: ExperimentResult, metrics: ExperimentMetrics):
    domain = np.arange(result.iterations)
    log_likelihood = metrics.log_likelihood

    with SubplotsAndSave(out_dir, 'log-likelihood', figsize = figsize(1, 1)) as (fig, ax):
        ax.plot(domain, log_likelihood[:result.iterations], label = 'Log-Likelihood')
        ax.set_title('Log-Likelihood (%s), %d Iterations' % (config.title, result.iterations))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Log-Likelihood')
        ax.legend(loc = 'lower right')
