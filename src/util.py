from functools import reduce
from typing import Callable, List, Optional, Union

import numpy as np
import progressbar
import sacred.utils
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



class PlainNumberWidget(progressbar.Widget):
    def __init__(self, format: str, observable: Callable[[], Optional[float]]):
        self._format = format
        self._observable = observable
        self._placeholder = ' ' * len(self._format % 0.0)


    def update(self, pbar):
        value = self._observable()
        return self._placeholder if value is None else self._format % value



class PlaceholderWidget(PlainNumberWidget):
    def __init__(self, format: str):
        super().__init__(format, observable = lambda: None)



class ExperimentNotConfiguredInterrupt(SacredInterrupt):
    STATUS = 'EXPERIMENT_NOT_CONFIGURED'



class MatrixProblemInterrupt(SacredInterrupt):
    STATUS = 'MATRIX_PROBLEM'



def bw_image(image: np.ndarray) -> np.ndarray:
    image[image == 255] = 1
    image[image == 0] = -1
    return image



def apply_sacred_frame_error_workaround() -> None:
    """
    Applies a workaround to ignore the KeyError thrown in ``sacred/utils.py:490``. Just treats
    every frame as a non-sacred frame (causing the exception traces to be a bit cluttered, but
    that's better than having the whole exception sucked up by the key error).
    """

    sacred.utils._is_sacred_frame = lambda frame: False



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



def symmetric(a: np.ndarray) -> torch.Tensor:
    return (a + a.T) / 2.0



def mlib_square(ax: Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # noinspection PyTypeChecker
    ax.set_aspect((x1 - x0) / (y1 - y0))



def build_dynamic_model(description: Union[str, List[str]], in_features: int, out_features: int) -> torch.nn.Module:
    prefix = torch.nn.__name__ + '.'
    p_globals = { name: eval(prefix + name) for name in dir(torch.nn) if name[0].isupper() }
    p_locals = { 'in_features': in_features, 'out_features': out_features }
    if type(description) == str:
        return eval(description, p_globals, p_locals)
    else:
        return eval('Sequential(%s)' % ', '.join(description), p_globals, p_locals)
