from typing import List, Tuple

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
        (N, p, T) = self._y.shape
        K = len(self._pi1)
        tiny = np.exp(-700)
        I = np.eye(K)
        const = (2 * np.pi) ** (-p / 2)
        problem = 0
        lik = 0

        Xcur = np.zeros((N, K, T))
        Xfin = np.zeros((N, K, T))

        Ppre = np.zeros((K, K, T))
        Pcur = np.zeros((K, K, T))
        Pfin = np.zeros((K, K, T))

        J = np.zeros((K, K, T))

        #
        # FORWARD PASS

        R = self._R + (self._R == 0) * tiny
        Xpre = np.ones((N, 1)) @ self._pi1.T
        Ppre[:, :, 0] = self._V1
        invR = np.diag(1 / self._R)
        for t in range(0, T):
            if K < p:
                temp1 = self._C / self._R.reshape(-1, 1)
                temp2 = temp1 @ Ppre[:, :, t]
                temp3 = self._C.T @ temp2
                temp4 = np.linalg.solve(I + temp3, temp1.T)
                invP = invR - temp2 @ temp4
                CP = temp1.T - temp3 @ temp4
            else:
                temp1 = np.diag(self._R) + self._C @ Ppre[:, :, t] @ self._C.T
                invP = np.linalg.inv(temp1)
                CP = self._C.T @ invP

            Kcur = Ppre[:, :, t] @ CP
            KC = Kcur @ self._C
            Ydiff = self._y[:, :, t] - Xpre @ self._C.T
            Xcur[:, :, t] = Xpre + Ydiff @ Kcur.T
            Pcur[:, :, t] = Ppre[:, :, t] - KC @ Ppre[:, :, t]

            if t < T - 1:
                Xpre = Xcur[:, :, t] @ self._A.T
                Ppre[:, :, t + 1] = self._A @ Pcur[:, :, t] @ self._A.T + self._Q

            detP = np.linalg.det(invP)
            if detP > 0:
                detiP = np.sqrt(detP)
                lik = lik + N * np.log(detiP) - 0.5 * np.sum(np.sum(np.multiply(Ydiff, Ydiff @ invP), axis = 0), axis = 0)
            else:
                problem = 1

        lik = lik + N * T * np.log(const)

        #
        # BACKWARD PASS

        A1 = np.zeros((K, K))
        t = T - 1
        Xfin[:, :, t] = Xcur[:, :, t]
        Pfin[:, :, t] = Pcur[:, :, t]
        Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N
        A2 = -Pt
        Ptsum = Pt

        YX = self._y[:, :, t].T @ Xfin[:, :, t]

        for t in reversed(range(0, T - 1)):
            J[:, :, t] = np.linalg.solve(Ppre[:, :, t + 1], self._A @ Pcur[:, :, t]).T
            Xfin[:, :, t] = Xcur[:, :, t] + (Xfin[:, :, t + 1] - Xcur[:, :, t] @ self._A.T) @ J[:, :, t].T

            Pfin[:, :, t] = Pcur[:, :, t] + J[:, :, t] @ (Pfin[:, :, t + 1] - Ppre[:, :, t + 1]) @ J[:, :, t].T
            Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N
            Ptsum = Ptsum + Pt
            YX = YX + self._y[:, :, t].T @ Xfin[:, :, t]

        A3 = Ptsum - Pt
        A2 = Ptsum + A2

        t = T - 1
        Pcov = (I - KC) @ self._A @ Pcur[:, :, t - 1]
        A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N

        for t in reversed(range(1, T - 1)):
            Pcov = (Pcur[:, :, t] + J[:, :, t] @ (Pcov - self._A @ Pcur[:, :, t])) @ J[:, :, t - 1].T
            A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N

        if problem:
            print('problem')
            problem = 0

        self._legacy = lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3


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
