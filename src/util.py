from functools import reduce
from typing import Optional

import torch
from matplotlib.axes import Axes
from sacred.utils import SacredInterrupt



class MatrixProblemInterrupt(SacredInterrupt):
    STATUS = 'MATRIX_PROBLEM'



def sum_ax0(a) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.sum(dim = 0)
    return reduce(lambda a, b: a + b, a)



def outer_batch(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if b is None:
        b = a
    return torch.einsum('bi,bj->bij', a, b)



def mlib_square(ax: Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # noinspection PyTypeChecker
    ax.set_aspect((x1 - x0) / (y1 - y0))
