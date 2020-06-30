import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src.em import EM
from src.util import MatrixProblemInterrupt


ex = Experiment('code')
ex.observers.append(FileStorageObserver('tmp_results'))



# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 42
    epsilon = 0.00001
    title = 'Simple Koopman with Polynomial Basis'
    T = 100
    N = 1
    h = 1
    latent_dim = 3
    param_mu = -0.1
    param_lambda = -0.05
    initial_value_x1 = 1.0
    initial_value_x2 = 1.0
    R = 1e-5 * np.eye(2)



@ex.capture
def sample_dynamics(T: int, N: int, h: float, param_mu: float, param_lambda: float, initial_value_x1: float, initial_value_x2: float, R: np.ndarray):
    ode = lambda x1, x2: np.array([param_mu * x1, param_lambda * (x2 - x1 ** 2)])

    sequences = []
    for _ in range(0, N):
        states = []
        for t in range(0, T):
            if t == 0:
                state = np.array((initial_value_x1, initial_value_x2))
            else:
                prev_state = states[-1]
                state = prev_state + h * ode(prev_state[0], prev_state[1])
            states.append(state)
        sequences.append(states)
    sequences = np.asarray(sequences)
    sequences += np.random.multivariate_normal(np.zeros(np.prod(sequences.shape)), sp.linalg.block_diag(*([R] * N * T))).reshape(sequences.shape)
    return np.asarray(sequences)



class Model(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        hidden = 100 * in_features * out_features
        self._pipe = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, out_features),
                torch.nn.ReLU()
        )


    def forward(self, x):
        return self._pipe(x)



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, epsilon: float, title: str, T: int, N: int, h: float, latent_dim: int):
    states = sample_dynamics()
    state_dim = states.shape[2]


    def callback(iteration, log_likelihood):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)


    em = EM(latent_dim, states, model = Model(latent_dim, state_dim))
    log_likelihoods = em.fit(epsilon, log = _log.info, callback = callback)
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

    fig, axs = plt.subplots(N, 1, sharex = 'all', figsize = (8, 4 * N), squeeze = False)
    with_label = True
    for sequence, ax in zip(range(N), axs.flatten()):
        for dim in range(state_dim):
            label = ('Dim. %d' % (dim + 1)) if with_label else None
            ax.plot(domain, states[sequence, :, dim], label = label)
        with_label = False
        ax.set_title('Sequence %d, States' % (sequence + 1))
        ax.set_ylabel('State')
        if sequence == N - 1:
            ax.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('Input States (%s)' % title)
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/input-states.png'
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
        ax.set_title('Sequence %d, Latent States' % (sequence + 1))
        ax.set_ylabel('Latent State')
        if sequence == N - 1:
            ax.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('Latent States (%s), %d Iterations' % (title, iterations))
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/latents.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    shutil.rmtree(out_dir)

    # Return the results.
    return {
            'iterations':     iterations,
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
