from typing import List, Optional, Tuple

import numpy as np



# noinspection PyPep8Naming


class EM:
    _state_dim: int
    _observation_dim: int
    _no_sequences: int

    _T: int
    _y: np.ndarray
    _yy: np.ndarray

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


    def __init__(self, state_dim: int, y: List[List[np.ndarray]]):
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

        # State dynamics matrix.
        self._A = np.eye(self._state_dim)
        # State noise covariance.
        self._Q = np.eye(self._state_dim)

        # Output matrix.
        self._C = np.eye(self._observation_dim, self._state_dim)
        # Output noise covariance.
        self._R = np.ones(self._observation_dim)

        # Initial state mean.
        self._pi1 = np.ones((self._state_dim, 1))
        # Initial state covariance.
        self._V1 = np.eye(self._state_dim)


    def fit(self, precision = 0.00001):
        history = []
        iteration = 0
        while True:
            # E STEP
            self.e_step()
            lik = self.get_likelihood()
            history.append(lik)
            print('cycle %d lik %f' % (iteration, lik))

            oldlik = history[-2] if iteration > 1 else None
            if iteration < 2:
                # Typically the first iteration of the EM-algorithm is far off, so set the likelihood base on the second iteration.
                likbase = lik
            elif lik < oldlik:
                print('violation')
            elif (lik - likbase) < (1 + precision) * (oldlik - likbase) or not np.isfinite(lik):
                print()
                break

            # M STEP
            self.m_step()

            iteration += 1

        return *self.get_estimations(), history, *self.get_problems()


    def e_step(self) -> None:
        I = np.eye(self._state_dim)
        problem = 0
        lik = 0

        #
        # Forward pass.

        P: List[Optional[np.ndarray]] = [None] * self._T
        V: List[Optional[np.ndarray]] = [None] * self._T
        m = np.zeros((self._no_sequences, self._state_dim, self._T))

        for t in range(0, self._T):
            if t > 0:
                m_pre = m[:, :, t - 1] @ self._A.T
                P_pre = self._A @ V[t - 1] @ self._A.T + self._Q
                P[t - 1] = P_pre
            else:
                # Initialization.
                m_pre = self._pi1.T
                P_pre = self._V1

            inv = np.linalg.inv(self._C @ P_pre @ self._C.T + np.diag(self._R))
            K = P_pre @ self._C.T @ inv
            y_diff = self._y[:, :, t] - m_pre @ self._C.T
            m[:, :, t] = m_pre + y_diff @ K.T
            V[t] = P_pre - K @ self._C @ P_pre

            detP = np.linalg.det(inv)
            if detP > 0:
                detiP = np.sqrt(detP)
                lik = lik + self._no_sequences * np.log(detiP) - 0.5 * np.sum(np.sum(np.multiply(y_diff, y_diff @ inv), axis = 0), axis = 0)
            else:
                problem = 1

        lik = lik + self._no_sequences * self._T * np.log((2 * np.pi) ** (-self._observation_dim / 2))

        #
        # Backward Pass.

        V_hat: List[Optional[np.ndarray]] = [None] * self._T
        J: List[Optional[np.ndarray]] = [None] * self._T
        m_hat = np.zeros((self._no_sequences, self._state_dim, self._T))
        self_correlation = []
        cross_correlation = []

        t = self._T - 1
        m_hat[:, :, t] = m[:, :, t]
        V_hat[t] = V[t]
        self_correlation.append(V_hat[t] + m_hat[:, :, t].T @ m_hat[:, :, t] / self._no_sequences)
        for t in reversed(range(1, self._T)):
            P_redone = P[t - 1]
            J[t - 1] = V[t - 1] @ self._A.T @ np.linalg.inv(P_redone)
            m_hat[:, :, t - 1] = m[:, :, t - 1] + (m_hat[:, :, t] - m[:, :, t - 1] @ self._A.T) @ J[t - 1].T
            V_hat[t - 1] = V[t - 1] + J[t - 1] @ (V_hat[t] - P_redone) @ J[t - 1].T

            self_correlation.append(V_hat[t - 1] + m_hat[:, :, t - 1].T @ m_hat[:, :, t - 1] / self._no_sequences)
            cross_correlation.append(J[t - 1] @ V_hat[t] + m_hat[:, :, t].T @ m_hat[:, :, t - 1] / self._no_sequences)  # Minka.

        # Ghahramani.
        t = self._T - 1
        Pcov = (I - K @ self._C) @ self._A @ V[t - 1]
        A1 = Pcov + m_hat[:, :, t].T @ m_hat[:, :, t - 1] / self._no_sequences
        for t in reversed(range(1, self._T - 1)):
            Pcov = (V[t] + J[t] @ (Pcov - self._A @ V[t])) @ J[t - 1].T
            A1 = A1 + Pcov + m_hat[:, :, t].T @ m_hat[:, :, t - 1] / self._no_sequences

        if problem:
            print('problem')

        Ptsum = np.sum(self_correlation, axis = 0)
        A1_new = np.sum(cross_correlation, axis = 0)
        A2 = Ptsum - self_correlation[0]
        A3 = Ptsum - self_correlation[-1]
        YX = np.sum([self._y[:, :, t].T @ m_hat[:, :, t] for t in range(self._T)], axis = 0)
        self._legacy = lik, m_hat, V_hat, Ptsum, YX, A1, A2, A3


    def m_step(self) -> None:
        _, Xfin, V_hat, Ptsum, YX, A1, A2, A3 = self._legacy

        self._pi1 = np.sum(Xfin[:, :, 0], axis = 0).reshape(1, -1).T / self._no_sequences
        T1 = Xfin[:, :, 0] - np.ones((self._no_sequences, 1)) @ self._pi1.T
        self._V1 = V_hat[0] + T1.T @ T1 / self._no_sequences
        self._C = np.linalg.solve(Ptsum, YX.T).T / self._no_sequences
        self._R = self._yy - np.diag(self._C @ YX.T) / (self._T * self._no_sequences)
        self._A = np.linalg.solve(A2, A1.T).T
        self._Q = (1 / (self._T - 1)) * np.diag(np.diag(A3 - self._A @ A1.T))

        self._Q_problem = np.linalg.det(self._Q) < 0
        self._R_problem = np.linalg.det(np.diag(self._R)) < 0
        self._V0_problem = np.linalg.det(self._V1) < 0


    def get_estimations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._A, self._Q, self._C, self._R, self._pi1, self._V1


    def get_likelihood(self) -> float:
        return self._legacy[0]


    def get_problems(self) -> Tuple[bool, bool, bool]:
        return self._Q_problem, self._R_problem, self._V0_problem
