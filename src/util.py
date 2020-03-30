import json

import numpy as np



class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)



def sample_linear_gaussian(T: int, pi1: np.ndarray, V1: np.ndarray, A: np.ndarray, Q: np.ndarray, C: np.ndarray, R: np.ndarray):
    xs = []
    ys = []
    for t in range(0, T):
        if t == 0:
            x = np.random.multivariate_normal(pi1, V1)
        else:
            x = np.random.multivariate_normal(A @ xs[-1], Q)
        y = np.random.multivariate_normal(C @ x, R)

        xs.append(x)
        ys.append(y)

    return xs, ys
