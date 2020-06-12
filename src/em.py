from typing import Callable, List, Optional, Tuple

import numpy as np

import torch
import torch.optim

from src import cubature
from src.util import outer_batch, sum_ax0


torch.set_default_dtype(torch.double)



class EM:
    _state_dim: int
    _observation_dim: int
    _no_sequences: int

    _T: int
    _y: np.ndarray
    _yy: np.ndarray

    _m_hat: np.ndarray

    _A: np.ndarray
    _Q: np.ndarray

    _C: np.ndarray
    _R: np.ndarray

    _m0: np.ndarray
    _V0: np.ndarray

    # Internal stuff.
    _P: np.ndarray
    _V: np.ndarray
    _m: np.ndarray
    _V_hat: np.ndarray
    _J: np.ndarray
    _self_correlation: np.ndarray
    _cross_correlation: np.ndarray

    _Q_problem: bool
    _C_problem: bool
    _V0_problem: bool


    def __init__(self, state_dim: int, y: List[List[np.ndarray]]):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._state_dim = state_dim
        self._observation_dim = y[0][0].shape[0]
        self._no_sequences = len(y)

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y[0])
        # Output vectors.
        # Normalize the measurements around the mean.
        y = np.array(y) - np.ones((self._no_sequences, self._T, 1)) @ np.mean(y, axis = (0, 1)).reshape(1, -1)
        self._y = np.transpose(y, axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]

        # Sum of the diagonal entries of the outer products y @ y.T.
        self._yy = np.sum(np.multiply(self._y, self._y), axis = (0, 2)).flatten() / (self._T * self._no_sequences)

        self._m_hat = np.zeros((self._no_sequences, self._state_dim, self._T))

        # State dynamics matrix.
        self._A = np.eye(self._state_dim)
        # State noise covariance.
        self._Q = np.eye(self._state_dim)

        # Output matrix.
        self._g = torch.nn.Linear(in_features = self._state_dim, out_features = self._observation_dim, bias = False)
        torch.nn.init.eye_(self._g.weight)
        # noinspection PyTypeChecker
        self._g = self._g.to(device = self._device)
        # Output noise covariance.
        self._R = np.ones(self._observation_dim)

        # Initial state mean.
        self._m0 = np.ones((self._state_dim, 1))
        # Initial state covariance.
        self._V0 = np.eye(self._state_dim)

        # Initialize matrices.
        self._y_hat = np.zeros((self._state_dim, self._no_sequences))
        self._P = np.zeros((self._state_dim, self._state_dim, self._T))
        self._V = np.zeros((self._state_dim, self._state_dim, self._T))
        self._m = np.zeros((self._no_sequences, self._state_dim, self._T))
        self._V_hat = np.zeros((self._state_dim, self._state_dim, self._T))
        self._J = np.zeros((self._state_dim, self._state_dim, self._T))
        self._self_correlation = np.zeros((self._state_dim, self._state_dim, self._T))
        self._cross_correlation = np.zeros((self._state_dim, self._state_dim, self._T))

        # Metrics for sanity checks.
        self._Q_problem: bool = False
        self._R_problem: bool = False
        self._V0_problem: bool = False

        self._optimizer_factory = lambda: torch.optim.Adagrad(params = self._g.parameters(), lr = 0.0001)


    def fit(self, precision = 0.00001, log: Callable[[str], None] = print, callback: Callable[[int, float], None] = lambda it, ll: None) -> List[float]:
        history = []
        likelihood_base = 0
        iteration = 0
        previous_likelihood = None
        while True:
            self.e_step()
            self.m_step()

            likelihood = self.calculate_likelihood()
            if likelihood is None:
                history.append(history[-1])
                log('Iter. %5d; Likelihood not computable.' % iteration)
            else:
                history.append(likelihood)
                log('Iter. %5d; Likelihood: %15.5f' % (iteration, likelihood))

            callback(iteration, likelihood)

            if likelihood is not None and previous_likelihood is not None and likelihood < previous_likelihood:
                log('Likelihood violation! New likelihood is higher than previous.')

            if iteration < 2:
                # Typically the first iteration of the EM-algorithm is far off, so set the likelihood base on the second iteration.
                likelihood_base = likelihood
            elif likelihood is not None and previous_likelihood is not None and (likelihood - likelihood_base) < (1 + precision) * (previous_likelihood - likelihood_base):
                log('Converged! :)')
                break

            previous_likelihood = likelihood
            iteration += 1
        return history


    def e_step(self) -> None:
        # TODO: Temporary treat g(.) as a linear function.
        C = next(self._g.parameters()).detach().cpu().numpy()

        #
        # Forward pass.

        K = np.zeros((self._state_dim, self._observation_dim))

        for t in range(0, self._T):
            if t == 0:
                # Initialization.
                m_pre = self._m0.T
                P_pre = self._V0
            else:
                m_pre = self._m[:, :, t - 1] @ self._A.T
                P_pre = self._A @ self._V[:, :, t - 1] @ self._A.T + self._Q
                self._P[:, :, t - 1] = P_pre

            innovation_cov = C @ P_pre @ C.T + np.diag(self._R)
            K = P_pre @ C.T @ np.linalg.inv(innovation_cov)
            y_diff = self._y[:, :, t] - m_pre @ C.T
            self._m[:, :, t] = m_pre + y_diff @ K.T
            self._V[:, :, t] = P_pre - K @ C @ P_pre

        #
        # Backward Pass.

        t = self._T - 1
        self._m_hat[:, :, t] = self._m[:, :, t]
        self._V_hat[:, :, t] = self._V[:, :, t]
        self._self_correlation[:, :, t] = self._V_hat[:, :, t] + self._m_hat[:, :, t].T @ self._m_hat[:, :, t] / self._no_sequences
        for t in reversed(range(1, self._T)):
            # J[t - 1] = V[t - 1] @ self._A.T @ np.linalg.inv(P[t - 1])
            self._J[:, :, t - 1] = np.linalg.solve(self._P[:, :, t - 1], self._A @ self._V[:, :, t - 1].T).T
            self._m_hat[:, :, t - 1] = self._m[:, :, t - 1] + (self._m_hat[:, :, t] - self._m[:, :, t - 1] @ self._A.T) @ self._J[:, :, t - 1].T
            self._V_hat[:, :, t - 1] = self._V[:, :, t - 1] + self._J[:, :, t - 1] @ (self._V_hat[:, :, t] - self._P[:, :, t - 1]) @ self._J[:, :, t - 1].T

            self._self_correlation[:, :, t - 1] = self._V_hat[:, :, t - 1] + self._m_hat[:, :, t - 1].T @ self._m_hat[:, :, t - 1] / self._no_sequences
            self._cross_correlation[:, :, t] = self._J[:, :, t - 1] @ self._V_hat[:, :, t] + self._m_hat[:, :, t].T @ self._m_hat[:, :, t - 1] / self._no_sequences  # Minka.


    def m_step(self) -> None:
        self._optimize_g()

        yx_sum = np.sum([self._y[:, :, t].T @ self._m_hat[:, :, t] for t in range(self._T)], axis = 0)
        self_correlation_sum = np.sum(self._self_correlation, axis = 2)
        cross_correlation_sum = np.sum(self._cross_correlation, axis = 2)
        # TODO: Temporary treat g(.) as a linear function.
        C = next(self._g.parameters()).detach().cpu().numpy()

        # self._C = np.linalg.solve(self_correlation_sum, yx_sum.T).T / self._no_sequences
        self._R = self._yy - np.diag(C @ yx_sum.T) / (self._T * self._no_sequences)
        # Do not subtract self._cross_correlation[0] here as there is no cross correlation \( P_{ 0, -1 } \) and thus it is not included in the list nor the sum.
        self._A = np.linalg.solve(self_correlation_sum - self._self_correlation[:, :, -1], cross_correlation_sum.T).T
        self._Q = np.diag(np.diag(self_correlation_sum - self._self_correlation[:, :, 0] - self._A @ cross_correlation_sum.T)) / (self._T - 1)
        self._m0 = np.mean(self._m_hat[:, :, 0], axis = 0).reshape(-1, 1)
        outer_part = self._m_hat[:, :, 0] - np.ones((self._no_sequences, 1)) @ self._m0.T
        self._V0 = self._self_correlation[:, :, 0] - np.outer(self._m0, self._m0) + outer_part.T @ outer_part / self._no_sequences

        self._Q_problem = np.linalg.det(self._Q) <= 0
        self._R_problem = np.linalg.det(np.diag(self._R)) <= 0
        self._V0_problem = np.linalg.det(self._V0) <= 0

        if self._Q_problem:
            print('Q problem!')
        if self._R_problem:
            print('R problem!')
        if self._V0_problem:
            print('V0 problem!')


    def _optimize_g(self):
        # TODO: Replace with numerical optimizer!
        yx_sum = np.sum([self._y[:, :, t].T @ self._m_hat[:, :, t] for t in range(self._T)], axis = 0)
        self_correlation_sum = np.sum(self._self_correlation, axis = 2)
        C_new = torch.tensor(np.linalg.solve(self_correlation_sum, yx_sum.T).T / self._no_sequences)
        # self._g.weight = torch.nn.Parameter(C_new, requires_grad = False)

        m_hat = torch.tensor(self._m_hat, dtype = torch.double, device = self._device)
        V_hat = torch.tensor(self._V_hat, dtype = torch.double, device = self._device)
        y = torch.tensor(self._y, dtype = torch.double, device = self._device)
        R = torch.tensor(self._R, dtype = torch.double, device = self._device).diag()

        estimate_g_hat = lambda n, t: cubature.spherical_radial_torch(self._state_dim, lambda x: self._g(x), m_hat[n, :, t], V_hat[:, :, t])[0]
        estimate_G = lambda n, t: cubature.spherical_radial_torch(self._state_dim, lambda x: outer_batch(self._g(x)), m_hat[n, :, t], V_hat[:, :, t])[0]

        optimizer = self._optimizer_factory()

        # Calculate the relevant parts of the expected log-likelihood only (increasing the computational performance).
        Q4_entry = lambda n, t: - (y[n, :, t].ger(estimate_g_hat(n, t)) @ R.inverse()).trace() \
                                - (estimate_g_hat(n, t).ger(y[n, :, t]) @ R.inverse()).trace() \
                                + (estimate_G(n, t) @ R.inverse()).trace()
        criterion_fn = lambda: sum_ax0([Q4_entry(n, t) for t in range(0, self._T) for n in range(0, self._no_sequences)])

        criterion = None
        criterion_prev = None
        while criterion_prev is None or (criterion - criterion_prev).abs() > 0.01:
            criterion = criterion_fn()

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            criterion_prev = criterion

        pass


    def _g_numpy(self, x: np.ndarray) -> np.ndarray:
        return self._g(torch.tensor(x, device = self._device)).detach().cpu().numpy()


    def calculate_likelihood(self) -> Optional[float]:
        if self._Q_problem or self._R_problem or self._V0_problem:
            return None

        # Store some variables to make the code below more readable.
        N = self._no_sequences
        p = self._observation_dim
        k = self._state_dim
        T = self._T
        A = self._A
        Q = self._Q
        m0 = self._m0.flatten()
        V0 = self._V0
        y = self._y
        m_hat = self._m_hat
        R = np.diag(self._R)

        q1 = -N * T * (k + p) * np.log(2.0 * np.pi) \
             - N * np.log(np.linalg.det(V0)) \
             - N * (T - 1) * np.log(np.linalg.det(Q)) \
             - N * T * np.log(np.linalg.det(R))

        V0_inverse = np.linalg.inv(V0)
        q2_entry = lambda n: (m_hat[n, :, 0] - m0).T @ (V0_inverse @ (m_hat[n, :, 0] - m0))
        q2 = -np.sum([q2_entry(n) for n in range(N)], axis = 0)

        Q_inverse = np.linalg.inv(Q)
        q3_entry = lambda n, t: (m_hat[n, :, t] - A @ m_hat[n, :, t - 1]).T @ (Q_inverse @ (m_hat[n, :, t] - A @ m_hat[n, :, t - 1]))
        q3 = -np.sum([q3_entry(n, t) for t in range(1, T) for n in range(N)], axis = 0)

        R_inverse = np.linalg.inv(R)
        q4_entry = lambda n, t: (y[n, :, t] - self._g_numpy(m_hat[n, :, t])).T @ (R_inverse @ (y[n, :, t] - self._g_numpy(m_hat[n, :, t])))
        q4 = -np.sum([q4_entry(n, t) for t in range(0, T) for n in range(N)], axis = 0)

        return (q1 + q2 + q3 + q4) / 2.0


    def get_estimated_states(self) -> np.ndarray:
        return self._m_hat


    def get_problems(self) -> Tuple[bool, bool, bool]:
        return self._Q_problem, self._R_problem, self._V0_problem
