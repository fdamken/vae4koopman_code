# Implementation of the spherical radial cubature rule proposed in Solin, Arno. “Cubature Integration Methods in Non-Linear Kalman Filtering and Smoothing,” 2010.
import math
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib_tuda
import numpy as np
import scipy.linalg
import torch

from src.util import mlib_square

matplotlib_tuda.load()
torch.set_default_dtype(torch.double)

_xi_cache = {'torch': {}, 'np': {}}


def spherical_radial(n: int, f: Callable[[np.ndarray], np.ndarray], mean: Optional[np.ndarray], cov: Optional[np.ndarray], cov_is_sqrt: bool = False,
                     cubature_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Computes the spherical radial cubature rule for gaussian probability density functions, i.e. the expectation value of ``f``
    with a gaussian distribution with mean ``mean`` and covariance matrix ``cov``.

    :param n: The dimension of the gaussian.
    :param f: Accepts shape ``(k, n)``, produces shape ``(k, m)``; The function to approximate the expectation of. Has to
              accept a batch ``k`` of points where axis ``0`` is the batch index and axis ``1`` is the dimensionality the
              same axis definitions must hold for the result.
    :param mean: Shape ``(b,n)``; The mean of the gaussian distribution. Axis ``0`` is the batch axis. Mutually exclusive with ``cubature_points``.
    :param cov: Shape ``(b, n, n)``; The covariance matrix of the gaussian distribution. Axis ``0`` is the batch axis. Mutually exclusive with ``cubature_points``.
    :param cov_is_sqrt: Sets whether the given ``cov`` is really a covariance matrix (``False``) or already is the principal
                        square root of the covariance matrix (``True``).
    :param cubature_points: Shape ``(2n, m)``; Overwrites the cubature points to use. Mutually exclusive with ``mean``/``cov``.
    :return: Tuple ``(result, cubature_points, cubature_points_transformed, L)``:
                 - ``result``: Shape matches the result of ``f`` along with axis ``0`` as the batch axis.; The approximation of the expectation value.
                 - ``cubature_points``: Shape ``(2n, n)``; The cubature points used for approximating the expected value.
                 - ``cubature_points_transformed``: Shape ``(2n, n)``; The cubature points used for approximating the expected value after the function evaluation.
                 - ``L``: The principal square root of ``cov``. Can be used to speed up other cubature evaluations.
    :return: Shape matches the result of ``f`` along with axis ``0`` as the batch axis.; The approximation of the expectation value.
    """

    return _spherical_radial(False, n, f, mean, cov, cov_is_sqrt, cubature_points)


def spherical_radial_torch(n: int, f: Callable[[torch.Tensor], torch.Tensor], mean: torch.Tensor, cov: torch.Tensor, cov_is_sqrt: bool = False,
                           cubature_points: Optional[np.ndarray] = None) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    PyTorch version of ``spherical_radial(..)``. See there for documentation.
    """

    return _spherical_radial(True, n, f, mean, cov, cov_is_sqrt, cubature_points)


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

    if n in _xi_cache['torch' if use_torch else 'np']:
        return _xi_cache['torch' if use_torch else 'np'][n]

    if use_torch:
        eye = torch.eye(n)
        result = torch.cat([eye, -eye], dim=0).to(device=device)
    else:
        eye = np.eye(n)
        result = np.concatenate([eye, -eye], axis=0)
    xi = math.sqrt(n) * result

    _xi_cache['torch' if use_torch else 'np'][n] = xi

    return xi


def _spherical_radial(use_torch: bool, n: int, f: Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]], mean: Optional[Union[np.ndarray, torch.Tensor]],
                      cov: Optional[Union[np.ndarray, torch.Tensor]], cov_is_sqrt: bool, cubature_points: Optional[np.ndarray] = None) \
        -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
    if cubature_points is None:
        if mean is None or cov is None:
            raise Exception('cubature_points is none but mean/cov are not both given!')
    else:
        if mean is not None or cov is not None:
            raise Exception('cubature_points is given but mean/cov are also given!')

    xi = _xi(use_torch, n, device=mean.device if use_torch else None)

    L = None
    if cubature_points is None:
        if cov_is_sqrt:
            L = cov
        else:
            cov_np = cov.detach().cpu().numpy() if use_torch else cov
            L_np = [scipy.linalg.sqrtm(it).astype(np.float) for it in cov_np]
            L = torch.tensor(L_np, dtype=cov.dtype, device=cov.device) if use_torch else np.asarray(L_np)

        if use_torch:
            cubature_points = torch.einsum('ij,bjk->bik', xi, L) + mean.unsqueeze(1)
        else:
            cubature_points = np.einsum('ij,bjk->bik', xi, L) + mean[:, np.newaxis, :]

    if use_torch:
        f_eval = f(cubature_points.reshape(-1, cubature_points.shape[2]))
        cubature_points_transformed = f_eval.view((cubature_points.shape[0], cubature_points.shape[1], *f_eval[0].shape))
        result_sum = cubature_points_transformed.sum(dim=1)
    else:
        f_eval = f(cubature_points.reshape(-1, cubature_points.shape[2]))
        cubature_points_transformed = f_eval.reshape((cubature_points.shape[0], cubature_points.shape[1], *f_eval[0].shape))
        # noinspection PyArgumentList
        result_sum = cubature_points_transformed.sum(axis=1)
    return result_sum / (2 * n), cubature_points, cubature_points_transformed, L


def _demo():
    b = 2
    n = 2
    mean_batch = np.array(b * [[80, 0.0]])
    cov_batch = np.asarray(b * [np.diag([40, 0.4])])
    mean_batch_t = torch.tensor(mean_batch)
    cov_batch_t = torch.tensor(cov_batch)

    f = lambda x: np.array([np.multiply(x[:, 0], np.cos(x[:, 1])),
                            np.multiply(x[:, 0], np.sin(x[:, 1]))]).T

    f_torch = lambda x: torch.cat([torch.mul(x[:, 0], torch.cos(x[:, 1])).view(-1, 1),
                                   torch.mul(x[:, 0], torch.sin(x[:, 1])).view(-1, 1)], dim=1)

    # NumPy version.
    approx_mean_batch_np, cubature_points_batch_np, cubature_points_transformed_batch_np, _ = spherical_radial(n, f, mean_batch, cov_batch)
    # PyTorch version.
    approx_mean_batch, cubature_points_batch, cubature_points_transformed_batch, _ = spherical_radial_torch(n, f_torch, mean_batch_t, cov_batch_t)
    approx_mean_batch = approx_mean_batch.numpy()
    cubature_points_batch = cubature_points_batch.numpy()
    # The above is calculated twice. This is useful for development to keep both implementations in sync.
    assert np.allclose(approx_mean_batch_np, approx_mean_batch)
    assert np.allclose(cubature_points_batch_np, cubature_points_batch)

    #
    # Plot the samples.
    for i, (mean, cov, approx_mean, cubature_points, cubature_points_transformed) in enumerate(
            zip(mean_batch, cov_batch, approx_mean_batch, cubature_points_batch, cubature_points_transformed_batch)):
        samples = np.random.multivariate_normal(mean, cov, 1000)
        samples_transformed = f(samples)
        monte_carlo_estimate = np.mean(samples_transformed, axis=0)

        # Original.
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(*samples.T, s=2, alpha=0.2, label='Samples', zorder=1)
        ax.scatter(*mean, marker='*', s=100, zorder=2, label='Mean')
        ax.scatter(*cubature_points.T, marker='+', zorder=3, label='Cubature Points')
        mlib_square(ax)
        ax.set_xlabel(r'$ r $')
        ax.set_ylabel(r'$ \theta $')
        ax.set_title('Original')
        ax.legend()
        fig.savefig('tmp_spherical-radial-cubature_%05d-original.pdf' % (i + 1))
        plt.close(fig)
        # Transformed.
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(*samples_transformed.T, s=1, alpha=0.2, zorder=1, label='Samples')
        ax.scatter(*approx_mean, marker='*', s=100, zorder=2, label='Approx. Mean')
        ax.scatter(*monte_carlo_estimate, marker='x', s=100, zorder=3, label='Monte Carlo Mean')
        ax.scatter(*cubature_points_transformed.T, marker='+', zorder=4, label='Cubature Points')
        mlib_square(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Transformed')
        ax.legend()
        fig.savefig('tmp_spherical-radial-cubature_%05d-transformed.pdf' % (i + 1))
        plt.close(fig)


if __name__ == '__main__':
    _demo()
