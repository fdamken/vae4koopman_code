from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim

from src import cubature
from src.util import outer_batch, outer_batch_torch


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

    _g: torch.nn.Module
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


    def __init__(self, state_dim: int, y: List[List[np.ndarray]], model: torch.nn.Module = None):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._state_dim = state_dim
        self._observation_dim = y[0][0].shape[0]
        self._no_sequences = len(y)

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y[0])
        # Output vectors.
        # Normalize the measurements around the mean.
        self._y = np.transpose(y, axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]

        # Sum of the diagonal entries of the outer products y @ y.T.
        self._yy = np.sum(np.multiply(self._y, self._y), axis = (0, 2)).flatten() / (self._T * self._no_sequences)

        self._m_hat = np.zeros((self._no_sequences, self._state_dim, self._T))

        # State dynamics matrix.
        self._A = np.eye(self._state_dim)
        # State noise covariance.
        self._Q = np.eye(self._state_dim)

        # Output matrix.
        if model is None:
            model = torch.nn.Linear(in_features = self._state_dim, out_features = self._observation_dim, bias = False)
            torch.nn.init.eye_(model.weight)
        self._g = model.to(self._device)
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

        self._optimizer = torch.optim.SGD(params = self._g.parameters(), lr = 0.001)


    def fit(self, precision: Optional[float] = 0.00001, /, max_iterations: int = np.inf, log: Callable[[str], None] = print,
            callback: Callable[[int, float], None] = lambda it, ll: None) -> List[
        float]:
        history = []
        likelihood_base = 0
        iteration = 0
        previous_likelihood = None
        while True:
            self.e_step()
            self.m_step()

            likelihood = self.calculate_likelihood()
            if likelihood is None:
                history.append(history[-1] if len(history) > 0 else -np.inf)
                log('Iter. %5d; Likelihood not computable.' % iteration)
            else:
                history.append(likelihood)
                log('Iter. %5d; Likelihood: %15.5f' % (iteration, likelihood))

            callback(iteration, likelihood)

            if likelihood is not None and previous_likelihood is not None and likelihood < previous_likelihood:
                log('Likelihood violation! New likelihood is lower than previous.')

            # Do not do convergence checking if no precision is set (this is useful for testing with a fixed no. of iterations).
            if precision is not None and likelihood is not None and previous_likelihood is not None:
                if np.abs(previous_likelihood - likelihood) < precision:
                    log('Converged! :)')
                    break

            previous_likelihood = likelihood
            iteration += 1

            if iteration >= max_iterations:
                log('Reached max. number of iterations: %d. Aborting!' % max_iterations)
                break
        return history


    def e_step(self) -> None:
        N = self._no_sequences
        k = self._state_dim

        #
        # Forward pass.

        for t in range(0, self._T):
            if t == 0:
                # Initialization.
                m_pre = self._m0.T.repeat(N, 0)
                P_pre = self._V0
            else:
                m_pre = self._m[:, :, t - 1] @ self._A.T
                P_pre = self._A @ self._V[:, :, t - 1] @ self._A.T + self._Q
                self._P[:, :, t - 1] = P_pre

            P_pre_batch = P_pre[np.newaxis, :, :].repeat(self._no_sequences, 0)
            y_hat = cubature.spherical_radial(k, lambda x: self._g_numpy(x), m_pre, P_pre_batch)[0]
            S = np.sum(cubature.spherical_radial(k, lambda x: outer_batch(self._g_numpy(x)), m_pre, P_pre_batch)[0], axis = 0) / N \
                - np.einsum('ni,nj->ij', y_hat, y_hat) / N + np.diag(self._R)
            S_inv = np.linalg.inv(S)
            K = np.sum(cubature.spherical_radial(k, lambda x: outer_batch(x, self._g_numpy(x)), m_pre, P_pre_batch)[0], axis = 0) / N \
                - np.einsum('ni,nj->ij', m_pre, y_hat) / N

            self._m[:, :, t] = m_pre + (self._y[:, :, t] - y_hat) @ S_inv.T @ K.T
            self._V[:, :, t] = P_pre - K @ S_inv @ K.T
            self._V[:, :, t] = (self._V[:, :, t] + self._V[:, :, t].T) / 2.0  # Numerical fixup of symmetry.

        #
        # Backward Pass.

        t = self._T - 1
        self._m_hat[:, :, t] = self._m[:, :, t]
        self._V_hat[:, :, t] = self._V[:, :, t]
        self._self_correlation[:, :, t] = self._V_hat[:, :, t] + self._m_hat[:, :, t].T @ self._m_hat[:, :, t] / self._no_sequences
        for t in reversed(range(1, self._T)):
            self._J[:, :, t - 1] = np.linalg.solve(self._P[:, :, t - 1], self._A @ self._V[:, :, t - 1].T).T
            self._m_hat[:, :, t - 1] = self._m[:, :, t - 1] + (self._m_hat[:, :, t] - self._m[:, :, t - 1] @ self._A.T) @ self._J[:, :, t - 1].T
            self._V_hat[:, :, t - 1] = self._V[:, :, t - 1] + self._J[:, :, t - 1] @ (self._V_hat[:, :, t] - self._P[:, :, t - 1]) @ self._J[:, :, t - 1].T
            self._V_hat[:, :, t - 1] = (self._V_hat[:, :, t - 1] + self._V_hat[:, :, t - 1].T) / 2.0  # Numerical fixup of symmetry.

            self._self_correlation[:, :, t - 1] = self._V_hat[:, :, t - 1] + self._m_hat[:, :, t - 1].T @ self._m_hat[:, :, t - 1] / self._no_sequences
            self._self_correlation[:, :, t - 1] = (self._self_correlation[:, :, t - 1] + self._self_correlation[:, :, t - 1].T) / 2.0  # Numerical fixup of symmetry.
            self._cross_correlation[:, :, t] = self._J[:, :, t - 1] @ self._V_hat[:, :, t] + self._m_hat[:, :, t].T @ self._m_hat[:, :, t - 1] / self._no_sequences  # Minka.


    def m_step(self) -> None:
        # self._optimize_g()

        self_correlation_sum = np.sum(self._self_correlation, axis = 2)
        cross_correlation_sum = np.sum(self._cross_correlation, axis = 2)

        g_hat, G = self._estimate_g_hat_and_G()
        g_hat = g_hat.detach().cpu().numpy()
        G = G.detach().cpu().numpy()

        self._R = (np.einsum('nit,nit->i', self._y, self._y)
                   - np.einsum('nti,nit->i', g_hat, self._y)
                   - np.einsum('nit,nti->i', self._y, g_hat)
                   + np.einsum('ntii->i', G)) / (self._no_sequences * self._T)

        # Do not subtract self._cross_correlation[0] here as there is no cross correlation \( P_{ 0, -1 } \) and thus it is not included in the list nor the sum.
        self._A = np.linalg.solve(self_correlation_sum - self._self_correlation[:, :, -1], cross_correlation_sum.T).T
        self._Q = np.diag(np.diag(self_correlation_sum - self._self_correlation[:, :, 0] - self._A @ cross_correlation_sum.T)) / (self._T - 1)
        self._m0 = np.mean(self._m_hat[:, :, 0], axis = 0).reshape(-1, 1)
        outer_part = self._m_hat[:, :, 0] - np.ones((self._no_sequences, 1)) @ self._m0.T
        self._V0 = self._self_correlation[:, :, 0] - np.outer(self._m0, self._m0) + outer_part.T @ outer_part / self._no_sequences

        self._Q_problem = not (np.linalg.eigvals(self._Q) >= 0).all()
        self._R_problem = not (np.linalg.eigvals(np.diag(self._R)) >= 0).all()
        self._V0_problem = not (np.linalg.eigvals(self._V0) >= 0).all()

        if self._Q_problem:
            print('Q problem!')
        if self._R_problem:
            print('R problem!')
        if self._V0_problem:
            print('V0 problem!')


    def _optimize_g(self) -> None:
        y = torch.tensor(self._y, dtype = torch.double, device = self._device)
        R = torch.tensor(self._R, dtype = torch.double, device = self._device).diag()
        R_inv = R.inverse()


        def criterion_fn():
            """
            Calculates the parts of the expected log-likelihood that are required for maximizing the LL w.r.t. the measurement
            parameters. That is, only \( Q_4 \) is calculated.

            Note that the sign of the LL is already flipped, such that the result of this function has to be minimized!
            """

            g_hat, G = self._estimate_g_hat_and_G()
            return - torch.einsum('nit,ntk,ki->', y, g_hat, R_inv) \
                   - torch.einsum('nti,nkt,ki->', g_hat, y, R_inv) \
                   + torch.einsum('ntik,ki->', G, R_inv)


        criterion = criterion_fn()
        self._optimizer.zero_grad()
        criterion.backward()
        self._optimizer.step()


    def _estimate_g_hat_and_G(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates \( \hat{\vec{g}} \) and \( \mat{G} \) in one go using the batch processing of the cubature rule implementation.

        :return: ``tuple(g_hat, G)``, where ``g_hat`` has the shape ``(N, T, p)`` and ``G`` has the shape ``(N, T, p, p)``.
        """

        m_hat = torch.tensor(self._m_hat, dtype = torch.double, device = self._device)
        V_hat = torch.tensor(self._V_hat, dtype = torch.double, device = self._device)

        m_hat_batch = m_hat.transpose(1, 2).reshape(-1, self._state_dim)
        V_hat_batch = torch.einsum('ijk->kij', V_hat).repeat(self._no_sequences, 1, 1)

        g_hat_batch = cubature.spherical_radial_torch(self._state_dim, lambda x: self._g(x), m_hat_batch, V_hat_batch)[0]
        G_batch = cubature.spherical_radial_torch(self._state_dim, lambda x: outer_batch_torch(self._g(x)), m_hat_batch, V_hat_batch)[0]

        g_hat = g_hat_batch.view((self._no_sequences, self._T, self._observation_dim))
        G = G_batch.view((self._no_sequences, self._T, self._observation_dim, self._observation_dim))
        return g_hat, G


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


    def get_estimations(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]:
        # noinspection PyTypeChecker
        return self._A, self._Q, self._g.state_dict(), self._R, self._m0, self._V0


    def get_estimated_states(self) -> np.ndarray:
        return self._m_hat


    def get_problems(self) -> Tuple[bool, bool, bool]:
        return self._Q_problem, self._R_problem, self._V0_problem
