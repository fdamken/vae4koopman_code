import re
from functools import reduce
from typing import Callable, List, Optional, Union

import numpy as np
import progressbar
import sacred.utils
import torch
import torch.nn
from matplotlib.axes import Axes
from sacred.utils import SacredInterrupt


class NumberTrendWidget(progressbar.Widget):
    def __init__(self, format_str: str, observable: Callable[[], Optional[float]]):
        self._format = format_str
        self._observable = observable
        self._placeholder = ' ' * (len(self._format % 0.0) + 2)
        self._previous = None

    def update(self, _):
        value = self._observable()
        if self._previous is not None and value is not None:
            if np.isclose(value, self._previous):
                suffix = ' \u25B6'  # Black right-pointing triangle.
            elif value > self._previous:
                suffix = ' \u25B2'  # Black up-pointing triangle.
            elif value < self._previous:
                suffix = ' \u25BC'  # Black down-pointing triangle.
            else:
                suffix = ' \u26A0'  # Warning sign.
        else:
            suffix = '  '
        self._previous = value
        return self._placeholder if value is None else (self._format % value + suffix)


class PlaceholderWidget(NumberTrendWidget):
    def __init__(self, format_str: str):
        super().__init__(format_str, lambda: None)


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
        return a.sum(dim=0)
    return reduce(lambda x, y: x + y, a)


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


def symmetric_batch(a: np.ndarray) -> torch.Tensor:
    return (a + a.transpose((0, 2, 1))) / 2.0


def ddiag(A: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(A))


def qr_batch(A: np.ndarray) -> np.ndarray:
    if not (2 <= len(A.shape) <= 3):
        raise Exception('A must be a matrix of a batch of matrices!')
    squeeze = False
    if len(A.shape) == 2:
        squeeze = True
        A = A[np.newaxis]
    R = np.asarray([np.linalg.qr(a, mode='complete')[1] for a in A])
    return R.squeeze(axis=0) if squeeze else R


def mlib_square(ax: Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # noinspection PyTypeChecker
    ax.set_aspect((x1 - x0) / (y1 - y0))


def random_from_descriptor(descriptor: str, size) -> np.ndarray:
    match = re.match(r'Random\.(\w+)\((.*)\)', descriptor)
    if match is None:
        raise Exception('Not a valid random descriptor!')
    method = match.group(1)
    params = [x.strip() for x in match.group(2).split(',') if x != '']
    if method == 'Uniform':
        if len(params) == 0:
            lower = 0.0
            upper = 1.0
        elif len(params) == 1:
            box = float(params[0])
            lower = -box
            upper = box
        elif len(params) == 2:
            lower = float(params[0])
            upper = float(params[1])
        else:
            raise Exception('Invalid number of parameters for uniform distribution: %d! Parameters: %s' % (len(params), str(params)))
        return np.random.uniform(lower, upper, size=size)
    raise Exception('Unknown drawing method %s!' % method)


def build_dynamic_model(description: Union[str, List[str]], in_features: int, out_features: int) -> torch.nn.Module:
    prefix = torch.nn.__name__ + '.'
    p_globals = {name: eval(prefix + name) for name in dir(torch.nn) if name[0].isupper()}
    p_locals = {'in_features': in_features, 'out_features': out_features}
    if type(description) == str:
        return eval(description, p_globals, p_locals)
    else:
        return eval('Sequential(%s)' % ', '.join(description), p_globals, p_locals)
