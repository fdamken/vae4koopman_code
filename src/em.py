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

        # Regularize the R matrix to not divide by zero.
        R = self._R + (self._R == 0) * np.exp(-700)

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

        self._pi1 = np.sum(self._x_hat[:, :, 0], axis = 0).reshape(1, -1).T / self._no_sequences
        T1 = self._x_hat[:, :, 0] - np.ones((int(self._no_sequences), 1)) @ self._pi1.T
        self._V1 = self._first_V_backward + T1.T @ T1 / self._no_sequences
        self._C = YX @ np.linalg.inv(self_correlation_sum) / self._no_sequences
        self._R = np.diag(YY - np.diag(self._C @ YX.T) / (self._T * self._no_sequences))
        self._A = cross_correlation_sum @ np.linalg.inv(self_correlation_sum - self._self_correlation[-1])
        self._Q = (1 / (self._T - 1)) * np.diag(np.diag(self_correlation_sum - self._self_correlation[0] - self._A @ cross_correlation_sum.T))

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
