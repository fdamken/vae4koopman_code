# Implementation of the spherical radial cubature rule proposed in Solin, Arno. “Cubature Integration Methods in Non-Linear Kalman Filtering and Smoothing,” 2010.
import math
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch

from src.util import mlib_square



def spherical_radial(n: int, f: Callable[[np.ndarray], np.ndarray], mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the spherical radial cubature rule for gaussian probability density functions, i.e. the expectation value of ``f``
    with a gaussian distribution with mean ``mean`` and covariance matrix ``cov``.

    :param n: The dimension of the gaussian.
    :param f: Accepts shape ``(2n, n)``, produces shape ``(2n, m)``; The function to approximate the expectation of. Has to
            accept a batch of points where axis ``0`` is the batch index and axis ``1`` is the dimensionality the same axis
            definitions must hold for the result.
    :param mean: Shape ``(b,n)``; The mean of the gaussian distribution. Axis ``0`` is the batch axis.
    :param cov: Shape ``(b, n, n)``; The covariance matrix of the gaussian distribution. Axis ``0`` is the batch axis.
    :return: Shape matches the result of ``f`` along with axis ``0`` as the batch axis.; The approximation of the expectation value.
    """

    return _spherical_radial(False, n, f, mean, cov)



def spherical_radial_torch(n: int, f: Callable[[torch.Tensor], torch.Tensor], mean: torch.Tensor, cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of ``spherical_radial(..)``. See there for documentation.
    """

    return _spherical_radial(True, n, f, mean, cov)



def _xi(use_torch: bool, n: int, device: Optional[torch.device]) -> Union[torch.Tensor, np.ndarray]:
    """
    Produces all cubature point vectors with each all zeros but at
    ``floor(i / 2)`` with the sign ``(-1) ** (i % 2)``, i.e. the ``i``-th
    intersection of an ``n``-dimensional unit sphere with the cartesian axes
    for each ``i``.

    :param use_torch: Whether to use the PyTorch implementation (``True``) or not (``False``).
    :param n: The dimension of a single vector.
    :return: Shape ``(2n, n)``; The cubature point vectors.
    """

    if use_torch:
        result = torch.zeros(2 * n, n, dtype = torch.float64)
        i = torch.arange(1, 2 * n + 1, dtype = torch.float64)
        # noinspection PyTypeChecker
        result[(i - 1).long(), ((i / 2).ceil() - 1).long()] = torch.tensor(-1, dtype = torch.float64) ** ((i - 1) % 2)
        result = result.to(device = device)
    else:
        result = np.zeros((2 * n, n))
        i = np.arange(1, 2 * n + 1, dtype = np.int)
        result[i - 1, (np.ceil(i / 2) - 1).astype(np.int)] = (-1) ** ((i - 1) % 2)
    return math.sqrt(n) * result



def _spherical_radial(use_torch: bool, n: int, f: Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]], mean: Union[np.ndarray, torch.Tensor],
                      cov: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    cov_np = cov.detach().cpu().numpy() if use_torch else cov
    L_np = [scipy.linalg.sqrtm(it).astype(np.float) for it in cov_np]
    L = torch.tensor(L_np, dtype = cov.dtype, device = cov.device) if use_torch else np.asarray(L_np)
    xi = _xi(use_torch, n, device = mean.device if use_torch else None)
    if use_torch:
        cubature_points = torch.einsum('ij,bjk->bik', xi, L) + mean.unsqueeze(1)
        f_eval = f(cubature_points.reshape(-1, mean.shape[1]))
        result_sum = f_eval.view((cubature_points.shape[0], cubature_points.shape[1], *f_eval[0].shape)).sum(dim = 1)
    else:
        cubature_points = np.einsum('ij,bjk->bik', xi, L) + mean[:, np.newaxis, :]
        f_eval = f(cubature_points.reshape(-1, mean.shape[1]))
        result_sum = f_eval.reshape((cubature_points.shape[0], cubature_points.shape[1], *f_eval[0].shape)).sum(axis = 1)
    return result_sum / (2 * n), cubature_points



def _demo():
    b = 5
    n = 2
    mean_batch = np.array(b * [[80, 0.0]])
    cov_batch = np.asarray(b * [np.diag([40, 0.4])])
    mean_batch_t = torch.tensor(mean_batch)
    cov_batch_t = torch.tensor(cov_batch)

    f = lambda x: np.array([np.multiply(x[:, 0], np.cos(x[:, 1])),
                            np.multiply(x[:, 0], np.sin(x[:, 1]))]).T

    f_torch = lambda x: torch.cat([torch.mul(x[:, 0], torch.cos(x[:, 1])).view(-1, 1),
                                   torch.mul(x[:, 0], torch.sin(x[:, 1])).view(-1, 1)], dim = 1)

    # NumPy version.
    approx_mean_batch_np, cubature_points_batch_np = spherical_radial(n, f, mean_batch, cov_batch)
    # PyTorch version.
    approx_mean_batch, cubature_points_batch = spherical_radial_torch(n, f_torch, mean_batch_t, cov_batch_t)
    approx_mean_batch = approx_mean_batch.numpy()
    cubature_points_batch = cubature_points_batch.numpy()
    # The above is calculated twice. This is useful for development to keep both implementations in sync.
    assert np.allclose(approx_mean_batch_np, approx_mean_batch)
    assert np.allclose(cubature_points_batch_np, cubature_points_batch)

    #
    # Plot the samples.
    for i, (mean, cov, approx_mean, cubature_points) in enumerate(zip(mean_batch, cov_batch, approx_mean_batch, cubature_points_batch)):
        samples = np.random.multivariate_normal(mean, cov, 1000)
        samples_transformed = f(samples)

        cubature_points_transformed = f(cubature_points)

        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (10, 5.5))
        if i == 0:
            s = 'st'
        elif i == 1:
            s = 'nd'
        elif i == 2:
            s = 'rd'
        else:
            s = 'th'
        fig.suptitle('%d%s Transformation of Gaussians' % (i + 1, s))
        # Original.
        ax1.scatter(*samples.T, s = 2, alpha = 0.2, label = 'Samples', zorder = 1)
        ax1.scatter(*mean, marker = '*', s = 100, zorder = 2, label = 'Mean')
        ax1.scatter(*cubature_points.T, marker = '+', zorder = 3, label = 'Cubature Points')
        mlib_square(ax1)
        ax1.set_xlabel(r'$ r $')
        ax1.set_ylabel(r'$ \theta $')
        ax1.set_title('Original')
        ax1.legend()
        # Transformed.
        ax2.scatter(*samples_transformed.T, s = 1, alpha = 0.2, zorder = 1, label = 'Samples')
        # ax2.scatter(*f(mean), marker = 'o', label = 'Naive Transformed Mean')
        ax2.scatter(*approx_mean, marker = '*', s = 100, zorder = 2, label = 'Approx. Mean')
        ax2.scatter(*cubature_points_transformed.T, marker = '+', zorder = 3, label = 'Cubature Points')
        # ax2.scatter(*monte_carlo_mean, marker = 'o', s = 25, zorder = 2, label = 'Monte Carlo Mean')
        mlib_square(ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Transformed')
        ax2.legend()
        # Configure ans show the figure.
        fig.savefig('tmp_spherical-radial-cubature_%05d.pdf' % (i + 1))
        fig.show()



if __name__ == '__main__':
    _demo()
