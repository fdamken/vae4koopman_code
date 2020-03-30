import json

import matplotlib.pyplot as plt
import numpy as np

from src.em import EM


np.random.seed(42)

EPSILON = 1e-5
EPSILON_ITERS = 10
PRINT_EVERY_N_ITERS = 1

EXAMPLES = {
        'Multi-Dimensional':                     {
                'enabled': True,
                'T':       5,
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
        'One-Dimensional State':                 {
                'enabled': False,
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
        'One-Dimensional State and Observation': {
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



class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)



def sample(T: int, pi1: np.ndarray, V1: np.ndarray, A: np.ndarray, Q: np.ndarray, C: np.ndarray, R: np.ndarray):
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

        states, observations = sample(T, pi1, V1, A, Q, C, R)
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

            x_prev_log_likelihood = log_likelihoods[-1] if log_likelihoods else None
            log_likelihoods.append(log_likelihood)

            if iteration % PRINT_EVERY_N_ITERS == 0:
                print('Iter. %d: pi1_loss: %.3f, V1_loss: %.3f, A_loss: %.3f, Q_loss: %.3f, C_loss: %.3f, R_loss: %.3f, x_loss: %.3f, log-likelihood: %.5f'
                      % (iteration, pi1_loss, V1_loss, A_loss, Q_loss, C_loss, R_loss, x_loss, log_likelihood))

            if x_prev_log_likelihood is not None and np.abs(log_likelihood - x_prev_log_likelihood) < EPSILON:
                if epsilon_iter < EPSILON_ITERS:
                    epsilon_iter += 1
                else:
                    print('Converged in %d iterations!' % iteration)
                    break
            else:
                epsilon_iter = 0

        plt.plot(np.arange(len(log_likelihoods)), log_likelihoods, label = 'Log-Likelihood')
        plt.title('Log-Likelihood (%s), %d Time steps' % (name, T))
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.legend()
        plt.savefig('tmp_%s-T%d-loglikelihood.png' % (name.replace(' ', '_'), T), dpi = 150)
        plt.show()

        if state_dim == 1:
            domain = np.arange(T)
            plt.plot(domain, states, label = 'True States')
            plt.plot(domain, x_est, label = 'Estimated States', linewidth = 1)
            if observations[0].shape[0] == 1:
                plt.scatter(domain, observations, label = 'Observations', s = 5, c = 'green')
            plt.title('States (%s), %d Iterations' % (name, iteration))
            plt.xlabel('Time Steps')
            plt.ylabel('State')
            plt.legend()
            plt.savefig('tmp_%s-T%d-states.png' % (name.replace(' ', '_'), T), dpi = 150)
            plt.show()

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
