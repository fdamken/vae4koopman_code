import json

import matplotlib.pyplot as plt
import numpy as np

from src.em import EM
from src.util import NumpyEncoder, sample_linear_gaussian


np.random.seed(42)

EPSILON = 1e-5
PRINT_EVERY_N_ITERS = 10

EXAMPLES = {
        '2D State, 2D Observation': {
                'enabled': False,
                'T':       10,
                'pi1':     np.array([0, 0]),
                'V1':      np.array([[1, 0],
                                     [0, 1]]),
                'A':       np.array([[1, 0],
                                     [0, 1]]),
                'Q':       np.array([[1, 0],
                                     [0, 1]]),
                'C':       np.array([[1, 0],
                                     [0, 1]]),
                'R':       np.array([[1, 0],
                                     [0, 1]])
        },
        '2D State, 1D Observation': {
                'enabled': False,
                'T':       10,
                'pi1':     np.array([0, 0]),
                'V1':      np.array([[1, 0],
                                     [0, 1]]),
                'A':       np.array([[1, 0],
                                     [0, 1]]),
                'Q':       np.array([[1, 0],
                                     [0, 1]]),
                'C':       np.array([[1, 0]]),
                'R':       np.array([[1]])
        },
        '1D State, 2D Observation': {
                'enabled': True,
                'T':       100,
                'pi1':     np.array([0]),
                'V1':      np.array([[1]]),
                'A':       np.array([[1]]),
                'Q':       np.array([[2]]),
                'C':       np.array([[1],
                                     [2]]),
                'R':       np.array([[2, 0],
                                     [0, 2]])
        },
        '1D State, 1D Observation': {
                'enabled': False,
                'T':       100,
                'pi1':     np.array([1]),
                'V1':      np.array([[1]]),
                'A':       np.array([[1]]),
                'Q':       np.array([[2]]),
                'C':       np.array([[1]]),
                'R':       np.array([[5]])
        }
}

if __name__ == '__main__':
    for name, params in EXAMPLES.items():
        enabled = params['enabled']
        T = params['T']
        pi1 = params['pi1']
        V1 = params['V1']
        A = params['A']
        Q = params['Q']
        C = params['C']
        R = params['R']

        if not params['enabled']:
            continue

        print(f'\n\n=== {name} ===\n')

        state_dim = pi1.shape[0]

        states, observations = sample_linear_gaussian(T, pi1, V1, A, Q, C, R)
        approximator = EM(state_dim, observations)

        print('pi1\n', pi1)
        print('V1\n', V1)
        print('A\n', A)
        print('Q\n', Q)
        print('C\n', C)
        print('R\n', R)
        print('states', states)
        print('observations', observations)

        # Perform two steps for testing.
        log_likelihoods = []
        iteration = 0
        epsilon_iter = 0
        while True:
            iteration += 1

            approximator.e_step()
            approximator.m_step()

            pi1_est, V1_est, A_est, Q_est, C_est, R_est, x_est, log_likelihood = approximator.get_estimations()
            pi1_loss = np.linalg.norm(pi1 - pi1_est)
            V1_loss = np.linalg.norm(V1 - V1_est)
            A_loss = np.linalg.norm(A - A_est)
            Q_loss = np.linalg.norm(Q - Q_est)
            C_loss = np.linalg.norm(C - C_est)
            R_loss = np.linalg.norm(R - R_est)
            x_loss = np.linalg.norm(np.array(states) - x_est)

            prev_log_likelihood = log_likelihoods[-1] if log_likelihoods else None
            log_likelihoods.append(log_likelihood)

            if iteration % PRINT_EVERY_N_ITERS == 0:
                print('%s; Iter. %d: pi1_loss: %.3f, V1_loss: %.3f, A_loss: %.3f, Q_loss: %.3f, C_loss: %.3f, R_loss: %.3f, x_loss: %.3f, log-likelihood: %.5f'
                      % (name, iteration, pi1_loss, V1_loss, A_loss, Q_loss, C_loss, R_loss, x_loss, log_likelihood))

            if prev_log_likelihood is not None:
                if log_likelihood < prev_log_likelihood:
                    raise Exception('New likelihood (%.5f) is lower than previous (%.5f)!' % (log_likelihood, prev_log_likelihood))

                if np.abs(log_likelihood - prev_log_likelihood) < EPSILON:
                    print('Converged in %d iterations!' % iteration)
                    break

        # Dump collected metrics.
        with open('tmp_%s-T%d.json' % (name.replace(' ', '_'), T), 'w') as file:
            print(json.dumps({
                    'iterations':     iteration,
                    'params':         params,
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
            }, cls = NumpyEncoder), file = file)

        #
        # Plot collected metrics.

        plt.plot(np.arange(len(log_likelihoods)), log_likelihoods, label = 'Log-Likelihood')
        plt.title('Log-Likelihood (%s), %d Time steps' % (name, T))
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.legend()
        plt.savefig('tmp_%s-loglikelihood.png' % name.replace(' ', '_'), dpi = 150)
        plt.show()

        domain = np.arange(T)
        states_array = np.array(states)
        x_est_array = np.array(x_est)
        for dim in range(state_dim):
            plt.plot(domain, states_array[:, dim].T, label = 'True States (Dim. %d)' % (dim + 1))
            plt.plot(domain, x_est_array[:, dim].T, label = 'Estimated States (Dim. %d)' % (dim + 1), linewidth = 1)
            # Only plot the observations if the state is 1D (otherwise the plot does not make sense).
            if state_dim == 1 and observations[dim].shape[0] == 1:
                plt.scatter(domain, observations, label = 'Observations', s = 5, c = 'green')
        plt.title('States (%s), %d Iterations' % (name, iteration))
        plt.xlabel('Time Steps')
        plt.ylabel('State')
        plt.legend()
        plt.savefig('tmp_%s-states.png' % name.replace(' ', '_'), dpi = 150)
        plt.show()
