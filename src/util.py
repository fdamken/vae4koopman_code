from functools import reduce
from typing import Optional

import numpy as np
import torch
import torch.nn
from matplotlib.axes import Axes
from sacred.utils import SacredInterrupt



class WhitenedModel(torch.nn.Module):
    _pipe: torch.nn.Module
    _device: torch.device
    _pca_matrix: torch.tensor


    def __init__(self, pipe: torch.nn.Module, in_features: int, device: torch.device = torch.device('cpu')):
        super().__init__()

        self._pipe = pipe
        self._device = device
        self._pca_matrix = torch.eye(in_features, device = device)


    def forward(self, x):
        return self._pipe(self._pca_transform(x))


    def fit_pca(self, X: np.ndarray):
        X_normalized = X - np.mean(X, axis = 0)
        C = np.cov(X_normalized.T)
        U, S, _ = np.linalg.svd(C)
        pca_matrix = U @ np.diag(1.0 / np.sqrt(S)).T
        # self._pca_matrix = torch.tensor(pca_matrix)


    def _pca_transform(self, x: torch.tensor):
        return x @ self._pca_matrix



class MatrixProblemInterrupt(SacredInterrupt):
    STATUS = 'MATRIX_PROBLEM'



def sum_ax0(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.sum(dim = 0)
    return reduce(lambda a, b: a + b, a)



def outer_batch(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    return np.einsum('bi,bj->bij', a, b)



def outer_batch_torch(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if b is None:
        b = a
    return torch.einsum('bi,bj->bij', a, b)



def mlib_square(ax: Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # noinspection PyTypeChecker
    ax.set_aspect((x1 - x0) / (y1 - y0))
