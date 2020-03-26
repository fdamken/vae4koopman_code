from typing import List

import numpy as np



def irange(frm, to):
    return range(frm, to + 1)



# noinspection PyPep8Naming
class EM:
    # TODO: Smart initialization would highly improve the convergence behavior.
    def __init__(self, state_dim: int, y: List[np.ndarray]):
        self._state_dim = state_dim
        self._out_dim = y[0].shape[0]

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y)
        # Output vectors.
        self._y = np.array(y)

        # State estimation.
        self._x_hat = [np.array(())] * self._T

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
        self._P = np.nan * np.ones((self._T, self._state_dim, self._state_dim))  # (lower) index
        self._P_backward = np.nan * np.ones((self._T, self._T, self._state_dim, self._state_dim))  # (lower1, lower2) index


    def e_step(self):
        # Push every dimension one out to get one-based indexing!
        x = np.nan * np.ones((self._T + 1, self._T + 1, self._state_dim))  # (lower, upper) index
        V = np.nan * np.ones((self._T + 1, self._T + 1, self._state_dim, self._state_dim))  # (lower, upper) index
        V_backward = np.nan * np.ones((self._T + 1, self._T + 1, self._T + 1, self._state_dim, self._state_dim))  # (lower1, lower2, upper) index
        J = np.nan * np.ones((self._T + 1, self._state_dim, self._state_dim))
        K = None

        # Forward iteration.
        for t in irange(1, self._T):
            if t == 1:
                # Initialize according to \( x_1^0 = \pi_1 \) and \( V_1^0 = V_1 \).
                x[t, t - 1, :] = self._pi1
                V[t, t - 1, :, :] = self._V1
            else:
                # Formulas (26), (27).
                x[t, t - 1, :] = self._A @ x[t - 1, t - 1, :]
                V[t, t - 1, :, :] = self._A @ V[t - 1, t - 1, :, :] @ self._A.T + self._Q
            # Formulas (28), (29), (30).
            K = V[t, t - 1, :, :] @ self._C.T @ np.linalg.inv(self._C @ V[t, t - 1, :, :] @ self._C.T + self._R)
            x[t, t, :] = x[t, t - 1, :] + K @ (self._y[t - 1] - self._C @ x[t, t - 1, :])
            V[t, t, :, :] = V[t, t - 1, :, :] - K @ self._C @ V[t, t - 1, :, :]

        # Backward iteration.
        for t in reversed(irange(1, self._T)):
            # Formulas (31), (32), (33).
            J[t - 1, :, :] = V[t - 1, t - 1, :, :] @ self._A.T @ np.linalg.inv(V[t, t - 1, :, :])
            x[t - 1, self._T, :] = x[t - 1, t - 1, :] + J[t - 1, :, :] @ (x[t, self._T, :] - self._A @ x[t - 1, t - 1, :])
            V[t - 1, self._T, :, :] = V[t - 1, t - 1, :, :] + J[t - 1, :, :] @ (V[t, self._T, :, :] - V[t, t - 1, :, :]) @ J[t - 1, :, :].T

        # Backward iteration two.
        V_backward[self._T, self._T - 1, self._T, :, :] = (np.eye(K.shape[0]) - K @ self._C) @ self._A @ V[self._T - 1, self._T - 1, :, :]
        for t in reversed(irange(2, self._T)):
            V_backward[t - 1, t - 2, self._T, :, :] = V[t - 1, t - 1, :, :] @ J[t - 2, :, :].T \
                                                      + J[t - 1, :, :] @ (V_backward[t, t - 1, self._T, :, :] - self._A @ V[t - 1, t - 1, :, :]) @ J[t - 2, :, :].T

        # Copy values to state and shift back to zero-based index.
        for t in irange(1, self._T):
            # Compute according to \( \hat{x}_t = x_t^T \), \( P_t = V_t^T + x_t^T (x_t^T)' \) and \( P_{t, t - 1} = V_{t, t - 1}^T + x_t^T (x_{t - 1}^T)' \).
            self._x_hat[t - 1] = x[t, self._T, :]
            self._P[t - 1, :, :] = V[t, self._T, :, :] + np.outer(x[t, self._T, :], x[t, self._T])
            self._P_backward[t - 1, t - 2, :, :] = V_backward[t, t - 1, self._T, :, :] + np.outer(x[t, self._T, :], x[t - 1, self._T, :])


    def m_step(self):
        # Formulas (14), (16), (18), (20).
        C_new = np.sum([np.outer(self._y[t], self._x_hat[t]) for t in range(self._T)], axis = 0) * np.linalg.inv(np.sum(self._P[0:self._T, :, :], axis = 0))
        R_new = np.sum([np.outer(self._y[t], self._y[t]) - C_new @ np.outer(self._x_hat[t], self._y[t]) for t in range(0, self._T)], axis = 0) / self._T
        A_new = np.sum([self._P_backward[t, t - 1, :, :] for t in range(1, self._T)], axis = 0) * np.linalg.inv(np.sum(self._P[0:(self._T - 1), :, :], axis = 0))
        # \( P_{t - 1, t} \) is not being calculated, but \( P_{t, t - 1} \). As \( x_t x_{t - 1}^T = x_{t - 1}^T x_t \) holds,
        # it is equivalent to use \( P_{t, t - 1} \) which is being calculated.
        Q_new = (np.sum(self._P[1:self._T, :, :], axis = 0) - A_new @ np.sum([self._P_backward[t, t - 1, :, :] for t in range(1, self._T)], axis = 0)) / (self._T - 1)
        # Formulas (22), (24).
        pi_new = self._x_hat[0]
        V_new = self._P[0, :, :] - np.outer(self._x_hat[0], self._x_hat[0])

        self._C = C_new
        self._R = R_new
        self._A = A_new
        self._Q = Q_new
        self._pi1 = pi_new
        self._V1 = V_new


    def get_estimations(self):
        return self._pi1, self._V1, self._A, self._Q, self._C, self._R, self._x_hat
