from typing import List, Optional, Tuple

import numpy as np



def irange(frm, to):
    return range(frm, to + 1)



# noinspection PyPep8Naming
class EM:
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


    # All the following formulas reference ones of the paper "From Hidden Markov Models to Linear Dynamical Systems" (Thomas P. Minka, 1999).
    def e_step(self) -> None:
        m = [None] * self._T
        V = [None] * self._T
        P = [None] * self._T
        V_hat = [None] * self._T

        # Forward recursion.
        # Formulas (56), (53), (54).
        K = np.linalg.solve(self._C @ self._V1 @ self._C.T + self._R, (self._V1 @ self._C.T).T).T
        m[0] = self._pi1 + K @ (self._y[0] - self._C @ self._pi1)
        V[0] = (np.eye(K.shape[0]) - K @ self._C) @ self._V1
        for t in range(1, self._T):
            # Formulas (49), (48), (50), (51).
            P[t - 1] = self._A @ V[t - 1] @ self._A.T + self._Q
            K = np.linalg.solve(self._C @ P[t - 1] @ self._C.T + self._R, (P[t - 1] @ self._C.T).T).T
            m[t] = self._A @ m[t - 1] + K @ (self._y[t] - self._C @ self._A @ m[t - 1])
            V[t] = (np.eye(K.shape[0]) - K @ self._C) @ P[t - 1]

        # Backward recursion.
        # Formulas (61), (62).
        self._x_hat[self._T - 1] = m[self._T - 1]
        V_hat[self._T - 1] = V[self._T - 1]
        self._P[self._T - 1] = V_hat[self._T - 1] + np.outer(m[self._T - 1], m[self._T - 1])
        self._P_backward[0] = np.array(0.0)
        for t in reversed(range(1, self._T)):
            # Formulas (58), (59), (60).
            J = np.linalg.solve(P[t - 1], (V[t - 1] @ self._A.T).T).T
            self._x_hat[t - 1] = m[t - 1] + J @ (self._x_hat[t] - self._A @ m[t - 1])
            V_hat[t - 1] = V[t - 1] + J @ (V_hat[t] - P[t - 1]) @ J.T

            self._P[t - 1] = V_hat[t - 1] + np.outer(self._x_hat[t - 1], self._x_hat[t - 1])
            self._P_backward[t] = J @ V_hat[t] + np.outer(self._x_hat[t], self._x_hat[t - 1])


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

        self._C = C_new
        self._R = R_new
        self._A = A_new
        self._Q = Q_new
        self._pi1 = pi_new
        self._V1 = V1_new


    def get_estimations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], float]:
        p = 1
        k = 1
        # @formatter:off
        log_likelihood = 0 \
            - np.sum([(self._y[t] - self._C @ self._x_hat[t]).T @ np.linalg.inv(self._R) @ (self._y[t] - self._C @ self._x_hat[t]) for t in range(0, self._T)]) / 2.0 \
            - self._T * np.log(np.linalg.det(self._R)) / 2.0 \
            - np.sum([(self._x_hat[t] - self._A @ self._x_hat[t - 1]).T @ np.linalg.inv(self._Q) @ (self._x_hat[t] - self._A @ self._x_hat[t - 1]) for t in range(1, self._T)]) / 2.0 \
            - (self._T - 1) * np.log(np.linalg.det(self._Q)) / 2.0 \
            - ((self._x_hat[0] - self._pi1[0]).T @ np.linalg.inv(self._V1) @ (self._x_hat[0] - self._pi1[0])) / 2.0 \
            - np.log(np.linalg.det(self._V1)) / 2.0 \
            - self._T * (p + k) * np.log(2 * np.pi) / 2.0
        # @formatter:on
        return self._pi1, self._V1, self._A, self._Q, self._C, self._R, self._x_hat, log_likelihood
