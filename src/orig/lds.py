import numpy as np

from src.orig.kalmansmooth import kalmansmooth



def lds(X, K = 2, T = None, cyc = 100, tol = 0.0001):
    p = len(X[0, :])
    N = len(X[:, 0])
    if T is None:
        T = N
    Mu = np.mean(X, axis = 0).reshape(1, -1)
    X = X - np.ones((N, 1)) @ Mu

    if N % T != 0:
        print('Error: Data matrix length must be multiple of sequence length T')
        return

    N = N / T

    # TODO: Initialization.
    A = np.eye(3)
    Q = np.ones(3)
    C = np.eye(3)
    R = np.ones(3)
    x0 = np.zeros(3).reshape(-1, 1)
    P0 = np.eye(3)

    likelihood = 0
    LL = []

    Y = X.reshape(int(T), int(N), int(p))
    Y = np.transpose(Y, axes = [1, 2, 0])

    YY = np.sum(np.multiply(X, X), axis = 0) / (T * N)

    for cycle in range(cyc):
        # E-Step.
        oldlik = likelihood
        likelihood, x_hat, V_backward, self_correlation, cross_correlation = kalmansmooth(A, C, Q, R, x0, P0, Y)
        LL.append(likelihood)
        print('cycle %d likelihood %f' % (cycle, likelihood))

        if cycle <= 2:
            likbase = likelihood
        elif likelihood < oldlik:
            print('violation')
        elif (likelihood - likbase) < (1 + tol) * (oldlik - likbase) or not np.isfinite(likelihood):
            print()
            break

        # M-Step.
        YX = np.sum([np.outer(Y[:, :, t], x_hat[:, :, t]) for t in range(T)], axis = 0)
        self_correlation_sum = np.sum(self_correlation, axis = 0)
        cross_correlation_sum = np.sum(cross_correlation, axis = 0)
        x0 = np.sum(x_hat[:, :, 0], axis = 0).reshape(1, -1).T / N
        T1 = x_hat[:, :, 0] - np.ones((int(N), 1)) @ x0.T
        P0 = V_backward[:, :, 0] + T1.T @ T1 / N
        C = YX @ np.linalg.inv(self_correlation_sum) / N
        R = YY - np.diag(C @ YX.T) / (T * N)
        A = cross_correlation_sum @ np.linalg.inv(self_correlation_sum - self_correlation[-1])
        Q = (1 / (T - 1)) * np.diag(np.diag(self_correlation_sum - self_correlation[0] - A @ cross_correlation_sum.T))
        if np.linalg.det(Q) < 0:
            print('Q problem')

    return { 'A': A, 'C': C, 'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'Mu': Mu, 'LL': LL }
