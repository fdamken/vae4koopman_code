import numpy as np

from src.em import LGDS_EM



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

    likelihood = 0
    LL = []
    ll = []

    Y = X.reshape(int(T), int(N), int(p))

    YY = np.sum(np.multiply(X, X), axis = 0) / (T * N)

    em = LGDS_EM(3, np.transpose(Y, axes = (1, 0, 2)), 3, 1, 100)
    for cycle in range(cyc):
        # E-Step.
        oldlik = likelihood
        # likelihood, x_hat, V_backward, self_correlation, cross_correlation = kalmansmooth(A, C, Q, R, x0, P0, Y)
        em.e_step()
        em.m_step()

        _, _, _, _, _, _, _, l = em.get_estimations()
        ll.append(l)
        likelihood = l

        print('cycle %d likelihood %f' % (cycle, likelihood))

        if cycle <= 2:
            likbase = likelihood
        elif likelihood < oldlik:
            print('violation')
        elif (likelihood - likbase) < (1 + tol) * (oldlik - likbase) or not np.isfinite(likelihood):
            print()
            break

    return LL, ll
