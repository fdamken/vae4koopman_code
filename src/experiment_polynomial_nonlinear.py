import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as sci
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.run import Run

from src.em import EM
from src.util import MatrixProblemInterrupt


ex = Experiment('code')
ex.observers.append(MongoObserver())
ex.observers.append(FileStorageObserver('tmp_results'))



# noinspection PyUnusedLocal
@ex.config
def config():
    title = 'Simple Koopman with Polynomial Basis'
    seed = 42
    # epsilon = 0.00001
    epsilon = None
    max_iterations = 5
    h = 0.02
    t_final = 1.0
    T = int(t_final / h)
    N = 1
    latent_dim = 10
    param_mu = -0.05
    param_lambda = -1.0
    initial_value_x1 = 0.3
    initial_value_x2 = 0.4
    R = 0.0 * np.eye(2)



@ex.capture
def sample_dynamics(t_final: float, T: int, N: int, h: float, param_mu: float, param_lambda: float, initial_value_x1: float, initial_value_x2: float, R: np.ndarray):
    ode = lambda x1, x2: np.asarray([param_mu * x1,
                                     param_lambda * (x2 - x1 ** 2)])

    initial_values = np.array((initial_value_x1, initial_value_x2))
    sequences = []
    for _ in range(0, N):
        sequences.append(sci.solve_ivp(lambda t, x: ode(*x), (0, t_final), initial_values, t_eval = np.arange(0, t_final, h), method = 'RK45').y.T)
    sequences = np.asarray(sequences)
    sequences += np.random.multivariate_normal(np.zeros(np.prod(sequences.shape)), sp.linalg.block_diag(*([R] * N * T))).reshape(sequences.shape)
    return sequences



class Model(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        hidden = 10 * in_features * out_features
        self._pipe = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden * 2),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden * 2, out_features),
                torch.nn.Tanh()
        )


    def forward(self, x):
        return self._pipe(x)



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, epsilon: float, max_iterations: int, title: str, T: int, N: int, h: float, latent_dim: int):
    observations = sample_dynamics()
    state_dim = observations.shape[2]


    def callback(iteration, log_likelihood):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)


    g = Model(latent_dim, state_dim)
    em = EM(latent_dim, observations, model = g)
    log_likelihoods = em.fit(epsilon, max_iterations = max_iterations, log = _log.info, callback = callback)
    A_est, Q_est, g_params_est, R_est, m0_est, V0_est = em.get_estimations()
    latents = em.get_estimated_states()

    # Collect results and metrics.
    Q_problem, R_problem, V0_problem = em.get_problems()
    final_log_likelihood = log_likelihoods[-1]
    iterations = len(log_likelihoods)

    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    #
    # Plot collected metrics, add to sacred and delete the plots afterwards.
    out_dir = tempfile.mkdtemp()

    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), log_likelihoods, label = 'Log-Likelihood')
    ax.set_title('Log-Likelihood (%s), %d Time steps' % (title, T))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.legend()
    out_file = f'{out_dir}/loglikelihood.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    domain = np.arange(T) * h

    latents_tensor = torch.tensor(latents, device = em._device)
    reconstructed_states = g(latents_tensor.transpose(1, 2).reshape(-1, latent_dim)).view((N, T, state_dim))
    reconstructed_states = reconstructed_states.detach().cpu().numpy()
    fig, axs = plt.subplots(N, 2, sharex = 'all', figsize = (10, 4 * N), squeeze = False)
    with_label = True
    for sequence, (ax1, ax2) in zip(range(N), axs):
        for dim in range(state_dim):
            label = ('Dim. %d' % (dim + 1)) if with_label else None
            ax1.plot(domain, observations[sequence, :, dim], label = label)
            ax2.plot(domain, reconstructed_states[sequence, :, dim])
        with_label = False
        ax1.set_title('Sequence %d, Input' % (sequence + 1))
        ax2.set_title('Sequence %d, Reconstructed' % (sequence + 1))
        ax1.set_ylabel('State')
        if sequence == N - 1:
            ax1.set_xlabel('Time Steps')
            ax2.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('Input and Reconstructed Observations (%s), %d Iterations' % (title, iterations))
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/observations.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    fig, axs = plt.subplots(N, 1, sharex = 'all', figsize = (8, 4 * N), squeeze = False)
    with_label = True
    for sequence, ax in zip(range(N), axs.flatten()):
        for dim in range(latent_dim):
            label = ('Dim. %d' % (dim + 1)) if with_label else None
            ax.plot(domain, latents[sequence, dim, :], label = label)
        with_label = False
        ax.set_title('Sequence %d' % (sequence + 1))
        ax.set_ylabel('Latents')
        if sequence == N - 1:
            ax.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('Latents (%s), %d Iterations' % (title, iterations))
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/latents.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    shutil.rmtree(out_dir)

    # Return the results.
    return {
            'iterations':     iterations,
            'input':          {
                    'observations': observations
            },
            'estimations':    {
                    'latents':  latents,
                    'A':        A_est,
                    'Q':        Q_est,
                    'g_params': g_params_est,
                    'R':        R_est,
                    'm0':       m0_est,
                    'V0':       V0_est
            },
            'log_likelihood': final_log_likelihood
    }
