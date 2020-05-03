import logging
import shutil
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src.em import LGDS_EM


ex = Experiment('lgds')
ex.observers.append(FileStorageObserver('tmp_results'))



# noinspection PyUnusedLocal
@ex.config
def config():
    seed = 42
    epsilon = 1e-5
    title = ''
    T = -1
    pi1 = np.array(0.0)
    V1 = np.array(0.0)
    A = np.array(0.0)
    Q = np.array(0.0)
    C = np.array(0.0)
    R = np.array(0.0)



@ex.capture
def sample_linear_gaussian(T: int, pi1: np.ndarray, V1: np.ndarray, A: np.ndarray, Q: np.ndarray, C: np.ndarray, R: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    xs = []
    ys = []
    for t in range(0, T):
        if t == 0:
            x = np.random.multivariate_normal(pi1, V1)
        else:
            x = np.random.multivariate_normal(A @ xs[-1], Q)
        y = np.random.multivariate_normal(C @ x, R)

        xs.append(x)
        ys.append(y)

    return xs, ys



# noinspection PyPep8Naming
@ex.main
def main(_run: Run, _log, epsilon, title, T, pi1, V1, A, Q, C, R):
    state_dim = pi1.shape[0]

    states, observations = sample_linear_gaussian()
    approximator = LGDS_EM(state_dim, [observations])

    # plt.scatter(*np.array(states).T, label = 'States')
    # plt.scatter(*np.array(observations).T, label = 'Observations')
    # plt.legend()
    # plt.title('Truth')
    # plt.show()

    _log.info('pi1: %s', str(pi1))
    _log.info('V1:  %s', str(V1).replace('\n', ''))
    _log.info('A:   %s', str(A).replace('\n', ''))
    _log.info('Q:   %s', str(Q).replace('\n', ''))
    _log.info('C:   %s', str(C).replace('\n', ''))
    _log.info('R:   %s', str(R).replace('\n', ''))
    _log.info('states:       %s', str(states))
    _log.info('observations: %s', str(observations))

    # Perform two steps for testing.
    log_likelihoods = []
    iteration = 0
    epsilon_iter = 0
    pis = []
    while True:
        iteration += 1

        approximator.e_step()
        approximator.m_step()

        pi1_est, V1_est, A_est, Q_est, C_est, R_est, x_est, log_likelihood = approximator.get_estimations()
        pis.append(pi1_est)
        pi1_loss = np.linalg.norm(pi1 - pi1_est)
        V1_loss = np.linalg.norm(V1 - V1_est)
        A_loss = np.linalg.norm(A - A_est)
        Q_loss = np.linalg.norm(Q - Q_est)
        C_loss = np.linalg.norm(C - C_est)
        R_loss = np.linalg.norm(R - R_est)
        x_loss = np.linalg.norm(np.transpose(np.array([states]), axes = (0, 2, 1)) - x_est)

        prev_log_likelihood = log_likelihoods[-1] if log_likelihoods else None
        log_likelihoods.append(log_likelihood)

        _run.log_scalar('p1_loss', pi1_loss, iteration)
        _run.log_scalar('V1_loss', V1_loss, iteration)
        _run.log_scalar('A_loss', A_loss, iteration)
        _run.log_scalar('Q_loss', Q_loss, iteration)
        _run.log_scalar('C_loss', C_loss, iteration)
        _run.log_scalar('R_loss', R_loss, iteration)
        _run.log_scalar('x_loss', x_loss, iteration)
        _run.log_scalar('log_likelihood', log_likelihood, iteration)

        _log.info('log_likelihood: %f', log_likelihood)

        if prev_log_likelihood is not None:
            # if log_likelihood < prev_log_likelihood:
            #    _log.error('New likelihood (%.5f) is lower than previous (%.5f)!' % (log_likelihood, prev_log_likelihood))
            #    raise LikelihoodDroppingInterrupt()

            if np.abs(log_likelihood - prev_log_likelihood) < epsilon:
                _log.info('Converged in %d iterations!' % iteration)
                break

    # piarray = np.array(pis)
    # plt.plot(np.arange(len(pis)), piarray[:, 0])
    # plt.plot(np.arange(len(pis)), piarray[:, 1])
    # plt.plot(np.arange(len(pis)), piarray[:, 2])
    # plt.title('Pis')
    # plt.show()

    #
    # Plot collected metrics, add to sacred and delete the plots afterwards.
    out_dir = tempfile.mkdtemp()

    plt.plot(np.arange(len(log_likelihoods)), log_likelihoods, label = 'Log-Likelihood')
    plt.title('Log-Likelihood (%s), %d Time steps' % (title, T))
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    out_file = f'{out_dir}/loglikelihood.png'
    plt.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close()

    domain = np.arange(T)
    states_array = np.array(states)
    x_est_array = np.array(x_est)
    for dim in range(state_dim):
        plt.plot(domain, states_array[:, dim].T, label = 'True States (Dim. %d)' % (dim + 1))
        plt.plot(domain, x_est_array[:, dim].T, label = 'Estimated States (Dim. %d)' % (dim + 1), linewidth = 1)
        # Only plot the observations if the state is 1D (otherwise the plot does not make sense).
        if state_dim == 1 and observations[dim].shape[0] == 1:
            plt.scatter(domain, observations, label = 'Observations', s = 5, c = 'green')
    plt.title('States (%s), %d Iterations' % (title, iteration))
    plt.xlabel('Time Steps')
    plt.ylabel('State')
    plt.legend()
    out_file = f'{out_dir}/states.png'
    plt.savefig(out_file, dpi = 150)
    _run.add_artifact(out_file)
    plt.close()

    shutil.rmtree(out_dir)

    # Return the results.
    return {
            'iterations':     iteration,
            'estimations':    {
                    'pi1': pi1_est,
                    'V1':  V1_est,
                    'A':   A_est,
                    'Q':   Q_est,
                    'C':   C_est,
                    'R':   R_est,
                    'x':   x_est
            },
            'losses':         {
                    'pi1': pi1_loss,
                    'V1':  V1_loss,
                    'A':   A_loss,
                    'Q':   Q_loss,
                    'C':   C_loss,
                    'R':   R_loss,
                    'x':   x_loss
            },
            'log_likelihood': log_likelihood
    }
