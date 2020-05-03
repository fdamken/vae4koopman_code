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
        self._x_hat = np.zeros((self._no_sequences, self._state_dim, self._T))
        V_hat: List[Optional[np.ndarray]] = [None] * self._T

        t = self._T - 1
        self._x_hat[:, :, t] = m[:, :, t]
        V_hat[t] = V[t]
        self_correlation = [V_hat[t] + self._x_hat[:, :, t].T @ self._x_hat[:, :, t] / self._no_sequences]
        for t in reversed(range(0, self._T - 1)):
            J[t] = V[t] @ self._A.T @ np.linalg.inv(P[t])
            self._x_hat[:, :, t] = m[:, :, t] + (self._x_hat[:, :, t + 1] - m[:, :, t] @ self._A.T) @ J[t].T

            V_hat[t] = V[t] + J[t] @ (V_hat[t + 1] - P[t]) @ J[t].T
            self_correlation.append(V_hat[t] + self._x_hat[:, :, t].T @ self._x_hat[:, :, t] / self._no_sequences)
        self._self_correlation = list(reversed(self_correlation))
        self._first_V_backward = V_hat[0]

        # Cross-correlation calculation according to Minka.
        cross_correlation_minka = []
        for t in reversed(range(1, self._T)):
            Pcov = J[t - 1] @ V_hat[t]
            cross_correlation_minka.append(Pcov + self._x_hat[:, :, t].T @ self._x_hat[:, :, t - 1] / self._no_sequences)
        self._cross_correlation = list(reversed(cross_correlation_minka))


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



# noinspection PyPep8Naming
class EM_OLD:
    _state_dim: int
    _out_dim: int

    _T: int
    _y: np.ndarray

    _x_hat: List[Optional[np.ndarray]]

    _A: np.ndarray
    _Q: np.ndarray

    _C: np.ndarray
    _R: np.ndarray

    _pi1: np.ndarray
    _V1: np.ndarray

    _P: List[Optional[np.ndarray]]
    _P_backward: List[Optional[np.ndarray]]


    def __init__(self, state_dim: int, y: List[np.ndarray]):
        self._state_dim = state_dim
        self._out_dim = y[0].shape[0]

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y)
        # Output vectors.
        self._y = np.array(y)

        # State estimation.
        self._x_hat = [None] * self._T

        # State dynamics matrix.
        self._A = np.eye(self._state_dim)
        # State noise covariance.
        self._Q = np.eye(self._state_dim)

        # Output matrix.
        self._C = np.eye(self._out_dim, self._state_dim)
        # Output noise covariance.
        self._R = np.eye(self._out_dim)

        # Initial state mean.
        self._pi1 = np.zeros((self._state_dim,))
        # Initial state covariance.
        self._V1 = np.eye(self._state_dim)

        # Expectations \( P_t = E[x_t x_t' | {y}] \) and \( P_{t, t - 1} = E[x_t x_{t - 1}' | {y}] \).
        self._P = [None] * self._T
        # Represents \( P_{t, t - 1} \) where \( t \) is the index.
        self._P_backward = [None] * self._T


    def e_step(self) -> None:
        # All the following formulas reference ones of the paper "From Hidden Markov Models to Linear Dynamical Systems" (Thomas P. Minka, 1999).

        m: List[Optional[np.ndarray]] = [None] * self._T
        V: List[Optional[np.ndarray]] = [None] * self._T
        P: List[Optional[np.ndarray]] = [None] * self._T
        J: List[Optional[np.ndarray]] = [None] * self._T
        V_hat: List[Optional[np.ndarray]] = [None] * self._T

        # Forward recursion.
        # Formulas (56), (53), (54).
        K = np.linalg.solve(self._C @ self._V1 @ self._C.T + self._R, (self._V1 @ self._C.T).T).T
        m[0] = self._pi1 + K @ (self._y[0] - self._C @ self._pi1)
        V[0] = self._V1 - K @ self._C @ self._V1
        for t in range(1, self._T):
            # Formulas (49), (48), (50), (51).
            P[t - 1] = self._A @ V[t - 1] @ self._A.T + self._Q
            K = np.linalg.solve(self._C @ P[t - 1] @ self._C.T + self._R, (P[t - 1] @ self._C.T).T).T
            m[t] = self._A @ m[t - 1] + K @ (self._y[t] - self._C @ self._A @ m[t - 1])
            V[t] = P[t - 1] - K @ self._C @ P[t - 1]

        # plt.scatter(*np.array(np.array(m)).T)
        # plt.title('Kalman-Filter Result')
        # plt.show()

        # Backward recursion.
        # Formulas (61), (62).
        self._x_hat[self._T - 1] = m[self._T - 1]
        V_hat[self._T - 1] = V[self._T - 1]
        self._P[self._T - 1] = V_hat[self._T - 1] + np.outer(m[self._T - 1], m[self._T - 1])
        P_backward_minka = self._P_backward.copy()
        P_backward_minka[0] = np.zeros((self._state_dim, self._state_dim))
        for t in reversed(range(1, self._T)):
            # Formulas (58), (59), (60).
            J[t - 1] = np.linalg.solve(P[t - 1], (V[t - 1] @ self._A.T).T).T
            self._x_hat[t - 1] = m[t - 1] + J[t - 1] @ (self._x_hat[t] - self._A @ m[t - 1])
            V_hat[t - 1] = V[t - 1] + J[t - 1] @ (V_hat[t] - P[t - 1]) @ J[t - 1].T

            # Combine (cross-) covariance \( J_t \hat{V}_t \) / \( V_{t - 1} with the mean/cross-mean to get the (cross-) correlation. This is
            # due to Minka (64).
            self._P[t - 1] = V_hat[t - 1] + np.outer(self._x_hat[t - 1], self._x_hat[t - 1])
            P_backward_minka[t] = J[t - 1] @ V_hat[t] + np.outer(self._x_hat[t], self._x_hat[t - 1])

        # START: Ghahramani.
        V_backward: List[Optional[np.ndarray]] = [None] * self._T
        V_backward[self._T - 1] = self._A @ V[self._T - 2] - K @ self._C @ self._A @ V[self._T - 2]
        for t in reversed(range(2, self._T)):
            # noinspection PyUnboundLocalVariable
            V_backward[t - 1] = V[t - 1] @ J[t - 2].T + J[t - 1] @ (V[t] - self._A @ V[t - 1]) @ J[t - 2].T

        P_backward_ghahramani = self._P_backward.copy()
        P_backward_ghahramani[0] = np.zeros((self._state_dim, self._state_dim))
        for t in range(1, self._T):
            P_backward_ghahramani[t] = V_backward[t] + np.outer(self._x_hat[t], self._x_hat[t - 1])
        # END: Ghahramani.

        self._P_backward = P_backward_minka

        # plt.scatter(*np.array(np.array(self._x_hat)).T)
        # plt.title('Kalman-Smoother Result')
        # plt.show()


    def m_step(self) -> None:
        # Formulas (14), (16), (18), (20).
        x_hat_array = np.array(self._x_hat)
        C_new = np.linalg.solve(np.sum(self._P, axis = 0), self._y.T.dot(x_hat_array).T).T
        R_new = (self._y.T.dot(self._y) - C_new @ x_hat_array.T.dot(self._y)) / self._T
        P_backward_sum = np.sum(self._P_backward[1:], axis = 0)
        A_new = np.linalg.solve(np.sum(self._P[:(self._T - 1)], axis = 0), P_backward_sum.T).T
        Q_new = (np.sum(self._P[1:], axis = 0) - A_new @ P_backward_sum) / (self._T - 1)
        # Formulas (22), (24).
        pi_new = self._x_hat[0]
        V1_new = self._P[0] - np.outer(self._x_hat[0], self._x_hat[0])

        determinant_error = np.array((np.linalg.det(R_new), np.linalg.det(Q_new), np.linalg.det(V1_new))) < 0
        if determinant_error.any():
            raise Exception('Non-positive determinant found! (R, Q, V1): ' + str(tuple(determinant_error)))

        self._C = C_new
        self._R = R_new
        self._A = A_new
        self._Q = Q_new
        self._pi1 = pi_new
        self._V1 = V1_new


    def get_estimations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], float]:
        # @formatter:off
        log_likelihood = 0 \
            - np.sum([(self._y[t] - self._C @ self._x_hat[t]).T @ np.linalg.inv(self._R) @ (self._y[t] - self._C @ self._x_hat[t]) for t in range(0, self._T)]) \
            - self._T * np.log(np.linalg.det(self._R)) \
            - np.sum([(self._x_hat[t] - self._A @ self._x_hat[t - 1]).T @ np.linalg.inv(self._Q) @ (self._x_hat[t] - self._A @ self._x_hat[t - 1]) for t in range(1, self._T)]) \
            - (self._T - 1) * np.log(np.linalg.det(self._Q)) \
            - ((self._x_hat[0] - self._pi1[0]).T @ np.linalg.inv(self._V1) @ (self._x_hat[0] - self._pi1[0])) \
            - np.log(np.linalg.det(self._V1)) \
            - self._T * (self._out_dim + self._state_dim) * np.log(2 * np.pi)
        # @formatter:on
        log_likelihood /= 2.0
        return self._pi1, self._V1, self._A, self._Q, self._C, self._R, self._x_hat, log_likelihood
