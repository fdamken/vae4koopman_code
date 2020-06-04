import shutil
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src.em import EM
from src.util import MatrixProblemInterrupt


ex = Experiment('lgds')
ex.observers.append(FileStorageObserver('tmp_results'))



# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 42
    epsilon = 0.00001
    title = ''
    T = -1
    N = 1
    A = np.array(0.0)
    Q = np.array(0.0)
    C = np.array(0.0)
    R = np.array(0.0)
    m0 = np.array(0.0)
    V0 = np.array(0.0)



@ex.capture
def sample_linear_gaussian(T: int, N: int, A: np.ndarray, Q: np.ndarray, C: np.ndarray, R: np.ndarray, m0: np.ndarray, V0: np.ndarray):
    sequences_x = []
    sequences_y = []
    for _ in range(N):
        xs = []
        ys = []
        for t in range(0, T):
            if t == 0:
                x = np.random.multivariate_normal(m0, V0)
            else:
                x = np.random.multivariate_normal(A @ xs[-1], Q)
            y = np.random.multivariate_normal(C @ x, R)

            xs.append(x)
            ys.append(y)
        sequences_x.append(xs)
        sequences_y.append(ys)
    return sequences_x, sequences_y



# noinspection PyPep8Naming
@ex.main
def main(_run: Run, _log, epsilon, title, T, N, A, Q, C, R, m0, V0):
    state_dim = m0.shape[0]

    states, observations = sample_linear_gaussian()
    states_array = np.transpose(np.array(states), axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]

    _log.debug('pi1\n', m0)
    _log.debug('V1\n', V0)
    _log.debug('A\n', A)
    _log.debug('Q\n', Q)
    _log.debug('C\n', C)
    _log.debug('R\n', R)
    _log.debug('states', states)
    _log.debug('observations', observations)


    def compute_losses():
        A_est, Q_est, C_est, R_est, m0_est, V0_est, x_est = em.get_estimations()
        A_loss = np.linalg.norm(A - A_est)
        Q_loss = np.linalg.norm(Q - Q_est)
        C_loss = np.linalg.norm(C - C_est)
        R_loss = np.linalg.norm(R - R_est)
        m0_loss = np.linalg.norm(m0 - m0_est)
        V0_loss = np.linalg.norm(V0 - V0_est)
        x_loss = np.linalg.norm(states_array - x_est)
        return A_loss, Q_loss, C_loss, R_loss, m0_loss, V0_loss, x_loss


    def callback(iteration, log_likelihood):
        A_loss, Q_loss, C_loss, R_loss, m0_loss, V0_loss, x_loss = compute_losses()
        _run.log_scalar('m0_loss', x_loss, iteration)
        _run.log_scalar('V0_loss', V0_loss, iteration)
        _run.log_scalar('A_loss', A_loss, iteration)
        _run.log_scalar('Q_loss', Q_loss, iteration)
        _run.log_scalar('C_loss', C_loss, iteration)
        _run.log_scalar('R_loss', R_loss, iteration)
        _run.log_scalar('x_loss', x_loss, iteration)
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)


    em = EM(state_dim, observations)
    log_likelihoods = em.fit(epsilon, log = _log.info, callback = callback)

    # Collect results and metrics.
    Q_problem, R_problem, V0_problem = em.get_problems()
    A_est, Q_est, C_est, R_est, m0_est, V0_est, x_est = em.get_estimations()
    A_loss, Q_loss, C_loss, R_loss, m0_loss, V0_loss, x_loss = compute_losses()
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

    generated_states, _ = sample_linear_gaussian(T = T * 2, N = 4, A = A_est, Q = Q_est, C = C_est, R = R_est, m0 = m0_est, V0 = V0_est)
    generated_states = np.transpose(np.array(generated_states), axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]
    domain = np.arange(T * 2)
    fig, axs = plt.subplots(2, 2, sharex = 'all', sharey = 'row', figsize = (10, 4))
    with_label = True
    for sequence, ax in enumerate(axs.flatten()):
        ax.axvline(T, ls = '--', color = 'red')
        for dim in range(state_dim):
            label = ('Dim. %d' % (dim + 1)) if with_label else None
            ax.plot(domain, generated_states[sequence, dim, :].T, label = label)
        with_label = False
        if sequence % 2 == 0:
            ax.set_ylabel('State')
        if sequence > 1:
            ax.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('Sampled States (%s)' % title)
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/sampled_states.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    domain = np.arange(T)
    fig, axes = plt.subplots(N, 2, sharex = 'all', sharey = 'row', figsize = (10, 4 * N), squeeze = False)
    with_label = True
    for sequence in range(N):
        ax1, ax2 = tuple(axes[sequence, :])
        for dim in range(state_dim):
            label = ('Dim. %d' % (dim + 1)) if with_label else None
            ax1.plot(domain, states_array[sequence, dim, :].T, label = label)
            ax2.plot(domain, x_est[sequence, dim, :].T)
        with_label = False
        ax1.set_title('Sequence %d, True States' % (sequence + 1))
        ax2.set_title('Sequence %d, Estimated States' % (sequence + 1))
        ax1.set_ylabel('State')
        if sequence == N - 1:
            ax1.set_xlabel('Time Steps')
            ax2.set_xlabel('Time Steps')
    fig.legend()
    fig.suptitle('States (%s), %d Iterations' % (title, iterations))
    plt.subplots_adjust(right = 0.85)
    out_file = f'{out_dir}/states.png'
    fig.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close(fig)

    shutil.rmtree(out_dir)

    # Return the results.
    return {
            'iterations':     iterations,
            'estimations':    {
                    'pi1': m0_est,
                    'V1':  V0_est,
                    'A':   A_est,
                    'Q':   Q_est,
                    'C':   C_est,
                    'R':   R_est,
                    'x':   x_est
            },
            'losses':         {
                    'm0': m0_loss,
                    'V0': V0_loss,
                    'A':  A_loss,
                    'Q':  Q_loss,
                    'C':  C_loss,
                    'R':  R_loss,
                    'x':  x_loss
            },
            'log_likelihood': final_log_likelihood
    }