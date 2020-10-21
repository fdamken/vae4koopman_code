import numpy as np


def cholesky_update(L: np.ndarray, x: np.ndarray, mode: str = '+', inplace: bool = False):
    n = x.shape[0]
    if not inplace:
        L = L.copy()
    x = x.copy()

    if mode == '+':
        mode_m = 1
    elif mode == '-':
        mode_m = -1
    else:
        raise Exception('Unknown mode %s!' % mode)

    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + mode_m * x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[k + 1:n, k] = (L[k + 1:n, k] + mode_m * s * x[k + 1:n]) / c
            x[k + 1:n] = c * x[k + 1:n] - s * L[k + 1:n, k]
    return L
