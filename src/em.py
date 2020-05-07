from typing import List, Optional, Tuple

import numpy as np

from src.util import InvalidCovarianceMatrixInterrupt



class LGDS_EM:
    _state_dim: int
    _observation_dim: int
    _no_sequences: int

    _T: int
    _y: np.ndarray

    _x_hat: np.ndarray

    _A: np.ndarray
    _Q: np.ndarray

    _C: np.ndarray
    _R: np.ndarray

    _pi1: np.ndarray
    _V1: np.ndarray

    _self_correlation: List[np.ndarray]
    _cross_correlation: List[np.ndarray]
    _first_V_backward: np.ndarray


    def __init__(self, state_dim: int, y: List[List[np.ndarray]], observation_dim = None, no_sequences = None, T = None):
        self._state_dim = state_dim
        self._observation_dim = y[0][0].shape[0] if observation_dim is None else observation_dim
        self._no_sequences = len(y) if no_sequences is None else no_sequences

        if self._state_dim < self._observation_dim:
            raise Exception('state_dim < observation_dim is not (yet) supported!')

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y[0]) if T is None else T
        # Output vectors.
        self._y = np.transpose(np.array(y), axes = (0, 2, 1))  # [sequence, dim, T]

        # State dynamics matrix.
        self._A = np.eye(self._state_dim)
        # State noise covariance.
        self._Q = np.eye(self._state_dim)

        # Output matrix.
        self._C = np.eye(self._observation_dim, self._state_dim)
        # Output noise covariance.
        self._R = np.eye(self._observation_dim)

        # Initial state mean.
        self._pi1 = np.zeros((self._state_dim, 1))
        # Initial state covariance.
        self._V1 = np.eye(self._state_dim)


    def e_step(self):
        #
        # Forward pass.
        m = np.zeros((self._no_sequences, self._state_dim, self._T))
        P: List[Optional[np.ndarray]] = [None] * self._T
        V: List[Optional[np.ndarray]] = [None] * self._T

        # Equations (56), (53), (54).
        K = self._V1 @ self._C.T @ np.linalg.inv(self._C @ self._V1 @ self._C.T + self._R)
        m[:, :, 0] = self._pi1.T + (self._y[:, :, 0] - self._pi1.T @ self._C.T) @ K.T
        V[0] = self._V1 - K @ self._C @ self._V1
        for t in range(1, self._T):
            # Equations (49), (48), (50), (51).
            P[t - 1] = self._A @ V[t - 1] @ self._A.T + self._Q
            K = P[t - 1] @ self._C.T @ np.linalg.inv(self._C @ P[t - 1] @ self._C.T + self._R)
            m[:, :, t] = m[:, :, t - 1] @ self._A.T + (self._y[:, :, t] - m[:, :, t - 1] @ self._A.T @ self._C.T) @ K.T
            V[t] = P[t - 1] - K @ self._C @ P[t - 1]

        #
        # Backward pass.
        J: List[Optional[np.ndarray]] = [None] * self._T
        V_hat: List[Optional[np.ndarray]] = [None] * self._T
        self._x_hat = np.zeros((self._no_sequences, self._state_dim, self._T))
        self_correlation = []
        cross_correlation = []

        t = self._T - 1
        # Equations (61), (62) and cross-correlation, eqn. (64).
        self._x_hat[:, :, t] = m[:, :, t]
        V_hat[t] = V[t]
        self_correlation.append(V_hat[t] + np.outer(self._x_hat[:, :, t], self._x_hat[:, :, t]) / self._no_sequences)
        for t in reversed(range(1, self._T)):
            # Equations (58), (59), (60).
            J[t - 1] = V[t - 1] @ self._A.T @ np.linalg.inv(P[t - 1])
            self._x_hat[:, :, t - 1] = m[:, :, t - 1] + (self._x_hat[:, :, t] - m[:, :, t - 1] @ self._A.T) @ J[t - 1].T
            V_hat[t - 1] = V[t - 1] + J[t - 1] @ (V_hat[t] - P[t - 1]) @ J[t - 1].T

            # Self- and cross-correlation, eqn. (64).
            self_correlation.append(V_hat[t - 1] + np.outer(self._x_hat[:, :, t - 1], self._x_hat[:, :, t - 1]) / self._no_sequences)
            cross_correlation.append(J[t - 1] @ V_hat[t] + np.outer(self._x_hat[:, :, t], self._x_hat[:, :, t - 1]) / self._no_sequences)
        self._self_correlation = list(reversed(self_correlation))
        self._cross_correlation = list(reversed(cross_correlation))
        self._first_V_backward = V_hat[0]


    def m_step(self) -> None:
        YX = np.sum([np.outer(self._y[:, :, t], self._x_hat[:, :, t]) for t in range(self._T)], axis = 0)
        YY = np.sum([np.multiply(self._y[:, :, t], self._y[:, :, t]) for t in range(self._T)], axis = 0).flatten() / (self._T * self._no_sequences)
        self_correlation_sum = np.sum(self._self_correlation, axis = 0)
        cross_correlation_sum = np.sum(self._cross_correlation, axis = 0)

        C_old = YX @ np.linalg.inv(self_correlation_sum) / self._no_sequences
        R_old = np.diag(YY - np.diag(C_old @ YX.T) / (self._T * self._no_sequences))
        A_old = cross_correlation_sum @ np.linalg.inv(self_correlation_sum - self._self_correlation[-1])
        Q_old = (1 / (self._T - 1)) * np.diag(np.diag(self_correlation_sum - self._self_correlation[0] - A_old @ cross_correlation_sum.T))
        pi1_old = np.sum(self._x_hat[:, :, 0], axis = 0).reshape(1, -1).T / self._no_sequences
        T1 = self._x_hat[:, :, 0] - np.ones((int(self._no_sequences), 1)) @ pi1_old.T
        V1_old = self._first_V_backward + T1.T @ T1 / self._no_sequences

        yx_sum = np.sum([np.outer(self._y[:, :, t], self._x_hat[:, :, t]) for t in range(self._T)], axis = 0)
        yy_sum = np.sum([np.outer(self._y[:, :, t], self._y[:, :, t]) for t in range(self._T)], axis = 0)
        self_correlation_sum = np.sum(self._self_correlation, axis = 0)
        cross_correlation_sum = np.sum(self._cross_correlation, axis = 0)

        C_new = yx_sum @ np.linalg.inv(self_correlation_sum)
        R_new = np.diag(np.diag((yy_sum - C_new @ yx_sum.T) / (self._T * self._no_sequences)))
        A_new = (cross_correlation_sum - self._cross_correlation[0]) @ np.linalg.inv(self_correlation_sum - self._self_correlation[-1])
        Q_new = np.diag(np.diag(((self_correlation_sum - self._self_correlation[0]) - A_new @ (cross_correlation_sum - self._cross_correlation[0]).T) / (self._T - 1)))
        pi1_new = np.mean(self._x_hat[:, :, 0], axis = 0).reshape(-1, 1)
        V1_new = self._self_correlation[0] - np.outer(pi1_new, pi1_new) + np.mean(np.outer(self._x_hat[:, :, 0] - pi1_new.T, self._x_hat[:, :, 0] - pi1_new.T), axis = 0)

        self._C = C_new
        self._R = R_new
        self._A = A_new
        self._Q = Q_new
        self._pi1 = pi1_new
        self._V1 = V1_new

        #assert np.allclose(C_old, C_new)  # holds
        #assert np.allclose(R_old, R_new)  # holds
        #assert np.allclose(A_old, A_new)  # fails
        #assert np.allclose(Q_old, Q_new)  # fails
        #assert np.allclose(pi1_old, pi1_new)  # holds
        #assert np.allclose(V1_old, V1_new)  # holds

        self._C = C_old
        self._R = R_old
        self._A = A_old
        self._Q = Q_old
        self._pi1 = pi1_old
        self._V1 = V1_old

        invalid_matrices = []
        if np.linalg.det(self._V1) < 0:
            invalid_matrices.append('V1')
        if np.linalg.det(self._R) < 0:
            invalid_matrices.append('R')
        if np.linalg.det(self._Q) < 0:
            invalid_matrices.append('Q')
        if invalid_matrices:
            raise InvalidCovarianceMatrixInterrupt(invalid_matrices)


    def get_estimations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        # @formatter:off
        log_likelihood = 0 \
            - np.sum([(self._y[:, :, t] - self._x_hat[:, :, t] @ self._C.T) @ np.linalg.inv(self._R) @ (self._y[:, :, t] - self._x_hat[:, :, t] @ self._C.T).T for t in range(0, self._T)]) \
            - self._no_sequences * self._T * np.log(np.linalg.det(self._R)) \
            - np.sum([(self._x_hat[:, :, t] - self._x_hat[:, :, t - 1] @ self._A.T) @ np.linalg.inv(self._Q) @ (self._x_hat[:, :, t] - self._x_hat[:, :, t - 1] @ self._A.T).T for t in range(1, self._T)]) \
            - self._no_sequences *(self._T - 1) * np.log(np.linalg.det(self._Q)) \
            - np.sum((self._x_hat[:, :, 0] - self._pi1.T) @ np.linalg.inv(self._V1) @ (self._x_hat[:, :, 0] - self._pi1.T).T) \
            - self._no_sequences * np.log(np.linalg.det(self._V1)) \
            - self._no_sequences * self._T * (self._observation_dim + self._state_dim) * np.log(2 * np.pi)
        # @formatter:on
        log_likelihood /= 2.0
        return self._pi1, self._V1, self._A, self._Q, self._C, self._R, self._x_hat, log_likelihood.item()
