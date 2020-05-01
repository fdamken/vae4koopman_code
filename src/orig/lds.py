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

    lik = 0
    LL = []

    Y = X.reshape(int(T), int(N), int(p))
    Y = np.transpose(Y, axes = [1, 2, 0])

    YY = np.sum(np.multiply(X, X), axis = 0) / (T * N)

    for cycle in range(cyc):
        # E-Step.
        oldlik = lik
        lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3 = kalmansmooth(A, C, Q, R, x0, P0, Y)
        LL.append(lik)
        print('cycle %d lik %f' % (cycle, lik))

        if cycle <= 2:
            likbase = lik
        elif lik < oldlik:
            print('violation')
        elif (lik - likbase) < (1 + tol) * (oldlik - likbase) or not np.isfinite(lik):
            print()
            break

        # M-Step.
        x0 = np.sum(Xfin[:, :, 0], axis = 0).reshape(1, -1).T / N
        T1 = Xfin[:, :, 0] - np.ones((int(N), 1)) @ x0.T
        P0 = Pfin[:, :, 0] + T1.T @ T1 / N
        C = YX @ np.linalg.inv(Ptsum) / N
        R = YY - np.diag(C @ YX.T) / (T * N)
        A = A1 @ np.linalg.inv(A2)
        Q = (1 / (T - 1)) * np.diag(np.diag(A3 - A @ A1.T))
        if np.linalg.det(Q) < 0:
            print('Q problem')

    return { 'A': A, 'C': C, 'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'Mu': Mu, 'LL': LL }
