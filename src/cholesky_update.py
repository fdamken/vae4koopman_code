import numpy as np



def cholesky_update(L: np.ndarray, x: np.ndarray, inplace: bool = False):
    n = x.shape[0]
    if not inplace:
        L = L.copy()
    x = x.copy()
    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[k + 1:n, k] = (L[k + 1:n, k] + s * x[k + 1:n]) / c
            x[k + 1:n] = c * x[k + 1:n] - s * L[k + 1:n, k]
    return L
