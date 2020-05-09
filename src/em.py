from typing import List, Optional, Tuple

import numpy as np

# noinspection PyPep8Naming
from src.orig.kalmansmooth import kalmansmooth



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
        y = np.array(y) - np.ones((self._no_sequences, 1)) @ np.mean(y, axis = 1).reshape(1, -1)
        self._y = np.transpose(y, axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]
        self._yy = np.sum(np.multiply(self._y, self._y), axis = 2).flatten() / (self._T * self._no_sequences)

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
        self._legacy = kalmansmooth(self._A, self._C, self._Q, self._R, self._pi1, self._V1, self._y)


    def m_step(self) -> None:
        _, Xfin, Pfin, Ptsum, YX, A1, A2, A3 = self._legacy

        self._pi1 = np.sum(Xfin[:, :, 0], axis = 0).reshape(1, -1).T / self._no_sequences
        T1 = Xfin[:, :, 0] - np.ones((self._no_sequences, 1)) @ self._pi1.T
        self._V1 = Pfin[:, :, 0] + T1.T @ T1 / self._no_sequences
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
