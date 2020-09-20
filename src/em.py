import collections
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import progressbar
import scipy
import torch
import torch.optim
from progressbar import Bar, ETA, Percentage

from src import cubature
from src.util import NumberTrendWidget, outer_batch, outer_batch_torch, PlaceholderWidget, symmetric, symmetric_batch



class EMInitialization:
    A: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    g: Union[None, collections.OrderedDict, Callable[[torch.nn.Module], torch.nn.Module]] = None
    R: Optional[np.ndarray] = None
    m0: Optional[np.ndarray] = None
    V0: Optional[np.ndarray] = None



class EMOptions:
    do_lgds: bool

    precision: Optional[float] = 0.00001
    max_iterations: Optional[int] = None

    g_optimization_learning_rate: float = 0.01
    g_optimization_precision: float = 1e-5
    g_optimization_max_iterations: Optional[int] = None

    log: Callable[[str], None] = print



class EM:
    LIKELIHOOD_FORMAT = '%21.5f'

    _options: EMOptions

    _do_lgds: bool
    _do_control: bool

    _latent_dim: int
    _observation_dim: int
    _control_dim: Optional[int]
    _no_sequences: int

    _T: int
    _y: np.ndarray
    _yy: np.ndarray
    _u: Optional[np.ndarray]

    _m_hat: np.ndarray
    _m_hat_sqrt: np.ndarray

    _A: np.ndarray
    _B: np.ndarray
    _Q: np.ndarray

    _g_model: torch.nn.Module
    _R: np.ndarray

    _m0: np.ndarray
    _V0: np.ndarray

    # Internal stuff.
    _P: np.ndarray
    _V: np.ndarray
    _m: np.ndarray
    _m_pre: np.ndarray
    _V_hat: np.ndarray
    _J: np.ndarray
    _self_correlation: np.ndarray
    _cross_correlation: np.ndarray
    _V_sqrt: np.ndarray
    _m_sqrt: np.ndarray
    _V_hat_sqrt: np.ndarray
    _Z: np.ndarray
    _D: np.ndarray

    _Q_problem: bool
    _R_problem: bool
    _V0_problem: bool


    def __init__(self, latent_dim: int, y: Union[List[List[np.ndarray]], np.ndarray], u: Optional[Union[List[List[np.ndarray]], np.ndarray]],
                 model: torch.nn.Module = None, initialization: EMInitialization = EMInitialization(), options = EMOptions()):
        """
        Constructs an instance of the expectation maximization algorithm described in the thesis.

        Invoke ``fit()`` to start learning.
        
        :param latent_dim: Dimensionality of the linear latent space. 
        :param y: Observations of the nonlinear observation space.
        :param u: Control inputs used to generate the observations.
        :param model: Learnable observation model.
        :param initialization: Initialization values for the EM-parameters.
        :param options: Various options to control the EM-behavior.
        """

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._options = options

        self._do_lgds = options.do_lgds
        self._do_control = u is not None

        self._latent_dim = latent_dim
        self._observation_dim = y[0][0].shape[0]
        self._control_dim = u[0][0].shape[0] if self._do_control else None
        self._no_sequences = len(y)

        # Number of time steps, i.e. number of output vectors.
        self._T = len(y[0])
        # Output vectors.
        self._y = np.transpose(y, axes = (0, 2, 1))  # from [sequence, T, dim] to [sequence, dim, T]

        # Sum of the diagonal entries of the outer products y @ y.T.
        self._yy = np.sum(np.multiply(self._y, self._y), axis = (0, 2)).flatten() / (self._T * self._no_sequences)

        # Control inputs.
        self._u = np.transpose(u, axes = (0, 2, 1)) if self._do_control else None  # from [sequence, T, dim] to [sequence, dim, T].

        self._m_hat = np.zeros((self._no_sequences, self._latent_dim, self._T))
        self._m_hat_sqrt = np.zeros((self._no_sequences, self._latent_dim, self._T))

        # State dynamics matrix.
        self._A = np.eye(self._latent_dim) if initialization.A is None else initialization.A
        # Control matrix.
        self._B = np.eye(self._latent_dim, self._control_dim) if self._do_control else None
        # State noise covariance.
        self._Q = np.eye(self._latent_dim) if initialization.Q is None else initialization.Q

        # Output network.
        self._g_model = model.to(device = self._device)
        if initialization.g is not None:
            if type(initialization.g) == collections.OrderedDict:
                self._g_model.load_state_dict(initialization.g)
            else:
                self._g_model = initialization.g(self._g_model)
        # Output noise covariance.
        self._R = np.eye(self._observation_dim) if initialization.R is None else initialization.R

        # Initial latent mean.
        self._m0 = np.ones((self._latent_dim,)) if initialization.m0 is None else initialization.m0
        # Initial latent covariance.
        self._V0 = np.eye(self._latent_dim) if initialization.V0 is None else initialization.V0

        # Check matrix and vectors initialization shapes.
        if self._A.shape != (self._latent_dim, self._latent_dim):
            raise Exception('A has invalid shape! Expected %s, but got %s!', (str((self._latent_dim, self._latent_dim))), str(self._A.shape))
        if self._do_control and self._B.shape != (self._latent_dim, self._control_dim):
            raise Exception('B has invalid shape! Expected %s, but got %s!', (str((self._latent_dim, self._control_dim))), str(self._B.shape))
        if self._Q.shape != (self._latent_dim, self._latent_dim):
            raise Exception('Q has invalid shape! Expected %s, but got %s!', (str((self._latent_dim, self._latent_dim))), str(self._Q.shape))
        if self._R.shape != (self._observation_dim, self._observation_dim):
            raise Exception('R has invalid shape! Expected %s, but got %s!', (str((self._observation_dim, self._observation_dim))), str(self._R.shape))
        if self._m0.shape != (self._latent_dim,):
            raise Exception('m0 has invalid shape! Expected %s, but got %s!', (str((self._latent_dim,))), str(self._m0.shape))
        if self._V0.shape != (self._latent_dim, self._latent_dim):
            raise Exception('V0 has invalid shape! Expected %s, but got %s!', (str((self._latent_dim, self._latent_dim))), str(self._V0.shape))

        # Fix matrix and vectors shapes for further processing.
        self._m0 = self._m0.reshape((self._latent_dim, 1))

        # Initialize internal matrices these will be overwritten.
        self._y_hat = np.zeros((self._latent_dim, self._no_sequences))
        self._P = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._V = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._m = np.zeros((self._no_sequences, self._latent_dim, self._T))
        self._m_pre = np.zeros((self._no_sequences, self._latent_dim, self._T))
        self._V_hat = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._J = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._self_correlation = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._cross_correlation = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))

        self._V_sqrt = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._m_sqrt = np.zeros((self._no_sequences, self._latent_dim, self._T))
        self._V_hat_sqrt = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._Z = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))
        self._D = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim, self._T))

        # Metrics for sanity checks.
        self._Q_problem: bool = False
        self._R_problem: bool = False
        self._V0_problem: bool = False

        self._optimizer_factory = lambda: torch.optim.Adam(params = self._g_model.parameters(), lr = self._options.g_optimization_learning_rate)


    def fit(self, callback: Callable[[int, float, float, int, List[float]], None] = lambda it, ll: None) -> List[float]:
        """
        Executes the expectation maximization algorithm, performs convergence checking and max. iterations checking, etc.

        :param callback: Invoked after every iteration and can be used to track progress, e.g. in TensorBoard or similar.
                         Parameters: ``iteration, likelihood, g_likelihood, g_iterations, g_ll_history``
                           - ``iteration``: Current iteration.
                           - ``likelihood``: Current log-likelihood.
                           - ``g_likelihood``: Final value of the log-likelihood parts affected by g after the g-optimization.
                           - ``g_iterations``: No. of iterations performed to optimize g.
                           - ``g_ll_history``: History of g log-likelihood values during the optimization.
        :return: History of the log-likelihoods per iteration.
        """

        history = []
        iteration = 1
        previous_likelihood = None
        while True:
            self.e_step()
            g_ll, g_iterations, g_ll_history = self.m_step()

            likelihood = self._calculate_likelihood()
            if likelihood is None:
                history.append(history[-1] if len(history) > 0 else -np.inf)
                # noinspection PyStringFormat
                self._log(f'Iter. %5d;  Likelihood not computable.  (G-LL: {EM.LIKELIHOOD_FORMAT},  G-Iters.: %5d)' % (iteration, g_ll, g_iterations))
            else:
                # noinspection PyStringFormat
                self._log(f'Iter. %5d;  Likelihood: {EM.LIKELIHOOD_FORMAT} (G-LL: {EM.LIKELIHOOD_FORMAT},  G-Iters.: %5d)' % (iteration, likelihood, g_ll, g_iterations))
                history.append(likelihood)

            callback(iteration, likelihood, g_ll, g_iterations, g_ll_history)

            if likelihood is not None and previous_likelihood is not None and likelihood < previous_likelihood:
                self._log('Likelihood violation! New likelihood is lower than previous.')

            if self._options.precision is not None and likelihood is not None and previous_likelihood is not None:
                if np.abs(previous_likelihood - likelihood) < self._options.precision:
                    self._log('Converged! :)')
                    break

            previous_likelihood = likelihood
            iteration += 1

            if self._options.max_iterations is not None and iteration > self._options.max_iterations:
                # noinspection PyStringFormat
                self._log('Reached max. number of iterations: %d. Aborting!' % self._options.max_iterations)
                break
        return history


    def e_step(self) -> None:
        """
        Executes the E-step of the expectation maximization algorithm.
        """

        N = self._no_sequences
        k = self._latent_dim

        Q_sqrt = np.linalg.cholesky(self._Q)
        R_sqrt = np.linalg.cholesky(self._R)

        #
        # Forward pass.

        bar = progressbar.ProgressBar(widgets = ['E-Step Forward:  ', Percentage(), ' ', Bar(), ' ', ETA(), ' ', PlaceholderWidget(EM.LIKELIHOOD_FORMAT)],
                                      maxval = self._T - 1).start()

        self._m[:, :, 0] = self._m0.T.repeat(N, 0)
        self._V[:, :, :, 0] = self._V0[np.newaxis, :, :].repeat(N, 0)
        self._m_sqrt[:, :, 0] = self._m0.T.repeat(N, 0)
        self._V_sqrt[:, :, :, 0] = np.linalg.cholesky(self._V0)[np.newaxis, :, :].repeat(N, 0)
        bar.update(0)
        for t in range(1, self._T):
            # Square-Root Filter.
            # TODO: Optimize for multiple sequences (if needed).
            mean_x = np.zeros((self._no_sequences, self._latent_dim))
            mean_z = np.zeros((self._no_sequences, self._observation_dim))
            S_sqrt = np.zeros((self._no_sequences, self._observation_dim, self._observation_dim))
            m_sqrt = np.zeros((self._no_sequences, self._latent_dim))
            V_sqrt = np.zeros((self._no_sequences, self._latent_dim, self._latent_dim))
            for n in range(self._no_sequences):
                # Form augmented mean and covariance.
                x_a = np.vstack([self._m[n, :, np.newaxis, t - 1], np.zeros((self._latent_dim + self._observation_dim, 1))])
                P_a = scipy.linalg.block_diag(self._V_sqrt[n, :, :, t - 1], Q_sqrt, R_sqrt)
                x_a_rep = x_a.repeat(2 * self._latent_dim + self._observation_dim, 1)
                # Calculate sigma points.
                Gamma = np.sqrt(self._latent_dim) * P_a
                sigma_a = np.block([x_a_rep, x_a_rep + Gamma, x_a_rep - Gamma])
                sigma_x = sigma_a[:self._latent_dim, :]
                sigma_u = sigma_a[self._latent_dim:2 * self._latent_dim, :]
                sigma_v = sigma_a[2 * self._latent_dim:2 * self._latent_dim + self._observation_dim, :]
                # Time update.
                sigma_x_transformed = np.einsum('ij,ib->ib', self._A, sigma_x) + sigma_u
                if self._do_control:
                    sigma_x_transformed += (self._B @ self._u[n, :, t - 1])[:, np.newaxis]
                sigma_z_transformed = self._g_numpy(sigma_x_transformed.T).T + sigma_v
                mean_x[n, :] = np.mean(sigma_x_transformed[:, 1:], axis = 1)
                mean_z[n, :] = np.mean(sigma_z_transformed[:, 1:], axis = 1)
                # Smoothing covariance and gain.
                res_x = (sigma_x - self._m[n, :, np.newaxis, t - 1]) / np.sqrt(2 * self._latent_dim)
                res_x_transformed = (sigma_x_transformed - mean_x[n, :, np.newaxis]) / np.sqrt(2 * self._latent_dim)
                res_s = np.block([[np.zeros_like(res_x_transformed[:, 0]).reshape(-1, 1), res_x_transformed[:, 1:]], [np.zeros_like(res_x[:, 0]).reshape(-1, 1), res_x[:, 1:]]])
                qr_s = np.linalg.qr(res_s.T, mode = 'complete')[1].T
                X_s = qr_s[:self._latent_dim, :self._latent_dim]
                Y_s = qr_s[self._latent_dim:self._latent_dim + self._latent_dim, :self._latent_dim]
                self._Z[n, :, :, t] = qr_s[self._latent_dim:self._latent_dim + self._latent_dim, self._latent_dim:self._latent_dim + self._latent_dim]
                self._D[n, :, :, t] = np.linalg.solve(X_s.T, Y_s.T).T
                # Measurement update.
                res_z = (sigma_z_transformed - mean_z[n, :, np.newaxis]) / np.sqrt(2 * self._latent_dim)
                res = np.block([[np.zeros_like(res_z[:, 0]).reshape(-1, 1), res_z[:, 1:]], [np.zeros_like(res_x_transformed[:, 0]).reshape(-1, 1), res_x_transformed[:, 1:]]])
                qr = np.linalg.qr(res.T, mode = 'complete')[1].T
                S_sqrt[n, :, :] = qr[:self._observation_dim, :self._observation_dim]
                Y = qr[self._observation_dim:self._observation_dim + self._latent_dim, :self._observation_dim]
                V_sqrt[n, :, :] = qr[self._observation_dim:self._observation_dim + self._latent_dim, self._observation_dim:self._observation_dim + self._latent_dim]
                K_sqrt = np.linalg.solve(S_sqrt[n, :, :].T, Y.T).T
                m_sqrt[n, :] = mean_x[n, :] + K_sqrt @ (self._y[n, :, t] - mean_z[n, :])

            # Time Update.
            if self._do_control:
                m_pre = self._m[:, :, t - 1] @ self._A.T + self._u[:, :, t - 1] @ self._B.T
            else:
                m_pre = self._m[:, :, t - 1] @ self._A.T
            P_pre = self._A @ self._V[:, :, :, t - 1] @ self._A.T + self._Q

            # Measurement Update.
            y_hat, _, _, P_pre_batch_sqrt = cubature.spherical_radial(k, lambda x: self._g_numpy(x), m_pre, P_pre)
            S = cubature.spherical_radial(k, lambda x: outer_batch(self._g_numpy(x)), m_pre, P_pre_batch_sqrt, True)[0] - outer_batch(y_hat) + self._R
            P = cubature.spherical_radial(k, lambda x: outer_batch(x, self._g_numpy(x)), m_pre, P_pre_batch_sqrt, True)[0] - outer_batch(m_pre, y_hat)
            K = np.linalg.solve(S.transpose((0, 2, 1)), P.transpose((0, 2, 1))).transpose((0, 2, 1))
            m = m_pre + np.einsum('bij,bj->bi', K, (self._y[:, :, t] - y_hat))
            V = symmetric_batch(P_pre - K @ S @ K.transpose((0, 2, 1)))

            mean_x_ok = np.allclose(mean_x, m_pre)
            mean_z_ok = np.allclose(mean_z, y_hat)
            S_ok = np.allclose(S_sqrt @ S_sqrt.transpose((0, 2, 1)), S)
            m_ok = np.allclose(m_sqrt, m)
            V_ok = np.allclose(V_sqrt @ V_sqrt.transpose((0, 2, 1)), V)
            print('(mean_x, mean_z, S, m, V): (%d, %d, %d, %d, %d)' % (mean_x_ok, mean_z_ok, S_ok, m_ok, V_ok))

            # Store results.
            self._m_pre[:, :, t] = m_pre
            self._P[:, :, :, t - 1] = P_pre
            self._m[:, :, t] = m
            self._V[:, :, :, t] = V

            self._m_sqrt[:, :, t] = m_sqrt
            self._V_sqrt[:, :, :, t] = V_sqrt

            bar.update(t)
        bar.finish()

        #
        # Backward Pass.

        bar = progressbar.ProgressBar(widgets = ['E-Step Backward: ', Percentage(), ' ', Bar(), ' ', ETA(), ' ', PlaceholderWidget(EM.LIKELIHOOD_FORMAT)],
                                      maxval = self._T - 1).start()

        t = self._T - 1
        self._m_hat[:, :, t] = self._m[:, :, t]
        self._V_hat[:, :, :, t] = self._V[:, :, :, t]
        self._m_hat_sqrt[:, :, t] = self._m_sqrt[:, :, t]
        self._V_hat_sqrt[:, :, :, t] = self._V_sqrt[:, :, :, t]
        self._self_correlation[:, :, :, t] = self._V_hat[:, :, :, t] + outer_batch(self._m_hat[:, :, t])
        bar.update(0)
        for t in reversed(range(1, self._T)):
            self._J[:, :, :, t - 1] = np.linalg.solve(self._P[:, :, :, t - 1], self._A @ self._V[:, :, :, t - 1].transpose((0, 2, 1))).transpose((0, 2, 1))
            self._m_hat[:, :, t - 1] = self._m[:, :, t - 1] + np.einsum('bij,bj->bi', self._J[:, :, :, t - 1], self._m_hat[:, :, t] - self._m_pre[:, :, t])
            self._V_hat[:, :, :, t - 1] = \
                self._V[:, :, :, t - 1] + self._J[:, :, :, t - 1] @ (self._V_hat[:, :, :, t] - self._P[:, :, :, t - 1]) @ self._J[:, :, :, t - 1].transpose((0, 2, 1))
            self._V_hat[:, :, :, t - 1] = symmetric_batch(self._V_hat[:, :, :, t - 1])

            self._self_correlation[:, :, :, t - 1] = symmetric_batch(self._V_hat[:, :, :, t - 1] + outer_batch(self._m_hat[:, :, t - 1]))
            self._cross_correlation[:, :, :, t] = self._J[:, :, :, t - 1] @ self._V_hat[:, :, :, t] + outer_batch(self._m_hat[:, :, t], self._m_hat[:, :, t - 1])  # Minka.

            bar.update(self._T - t)
        bar.finish()


    def m_step(self) -> Tuple[float, int, List[float]]:
        """
        Executes the M-step of the expectation maximization algorithm.

        :return: ``(g_ll, g_iterations, g_ll_history)`` The final objective value after optimizing ``g`` (i.e. the value of the expected log-likelihood that are affected
                  by ``g``), `g_ll``, the number of gradient descent iterations needed for the optimization, ``g_iterations`` and the history of objective values, ``g_ll_history``.
        """

        g_ll, g_iterations, g_ll_history = self._optimize_g()

        self_correlation_mean = self._self_correlation.mean(axis = 0)
        cross_correlation_mean = self._cross_correlation.mean(axis = 0)
        self_correlation_sum = self_correlation_mean.sum(axis = 2)
        cross_correlation_sum = cross_correlation_mean.sum(axis = 2)

        g_hat, G, _ = self._estimate_g_hat_and_G()
        g_hat = g_hat.detach().cpu().numpy()
        G = G.detach().cpu().numpy()

        self._R = (np.einsum('nit,njt->ij', self._y, self._y)
                   - np.einsum('nti,njt->ij', g_hat, self._y)
                   - np.einsum('nit,ntj->ij', self._y, g_hat)
                   + np.einsum('ntij->ij', G)) / (self._no_sequences * self._T)

        if self._do_control:
            M = lambda n, t: np.block([[self._cross_correlation[n, :, :, t], np.outer(self._m_hat[n, :, t], self._u[n, :, t - 1])]])
            W = lambda n, t: np.block([[self._self_correlation[n, :, :, t], np.outer(self._m_hat[n, :, t], self._u[n, :, t])],
                                       [np.outer(self._u[n, :, t], self._m_hat[n, :, t]), np.outer(self._u[n, :, t], self._u[n, :, t])]])
            C1 = np.sum([M(n, t) for n in range(self._no_sequences) for t in range(1, self._T)], axis = 0)
            C2 = np.sum([2 * symmetric(W(n, t - 1)) for n in range(self._no_sequences) for t in range(1, self._T)], axis = 0)
            C = np.linalg.solve(C2.T, 2 * C1.T).T
            A_new = C[:, :self._latent_dim]
            B_new = C[:, self._latent_dim:]
            Q_part = [self._self_correlation[n, :, :, t] - C @ M(n, t).T - M(n, t) @ C.T + C @ W(n, t - 1) @ C.T for n in range(self._no_sequences) for t in range(1, self._T)]
            Q_new = np.sum(Q_part, axis = 0) / (self._no_sequences * (self._T - 1))

            self._A = A_new
            self._B = B_new
            self._Q = Q_new
        else:
            # Do not subtract self._cross_correlation[0] here as there is no cross correlation \( P_{ 0, -1 } \) and thus it is not included in the list nor the sum.
            self._A = np.linalg.solve(self_correlation_sum - self_correlation_mean[:, :, -1], cross_correlation_sum.T).T
            self._Q = (self_correlation_sum - self_correlation_mean[:, :, 0] - self._A @ cross_correlation_sum.T) / (self._T - 1)
        self._m0 = self._m_hat[:, :, 0].mean(axis = 0).reshape(-1, 1)
        self._V0 = self_correlation_mean[:, :, 0] - np.outer(self._m0, self._m0) + outer_batch(self._m_hat[:, :, 0] - self._m0.T).mean(axis = 0)

        # As Q and R are the diagonal of diagonal matrices, there entries are already the eigenvalues.
        Q_eigvals = np.linalg.eigvals(self._Q)
        R_eigvals = np.linalg.eigvals(self._R)
        V0_eigvals = np.linalg.eigvals(self._V0)
        self._Q_problem = not (Q_eigvals >= 0).all()
        self._R_problem = not (R_eigvals >= 0).all()
        self._V0_problem = not (V0_eigvals >= 0).all()

        if self._Q_problem:
            print('Q problem!  Negative eigenvalues: %s' % str(Q_eigvals[Q_eigvals < 0]))
        if self._R_problem:
            print('R problem!  Negative eigenvalues: %s' % str(R_eigvals[R_eigvals < 0]))
        if self._V0_problem:
            print('V0 problem! Negative eigenvalues: %s' % str(V0_eigvals[V0_eigvals < 0]))

        return g_ll, g_iterations, g_ll_history


    def _optimize_g(self) -> Tuple[float, int, List[float]]:
        """
        Optimized the measurement function ``g`` using the optimizer stored in ``self._optimizer``.

        :return: ``(g_ll, g_iterations, g_ll_history)`` The final objective value after optimizing ``g`` (i.e. the value of the expected log-likelihood that are affected
                  by ``g``), `g_ll``, the number of gradient descent iterations needed for the optimization, ``g_iterations`` and the history of objective values, ``g_ll_history``.
        """

        y = torch.tensor(self._y, dtype = torch.double, device = self._device)
        R_inv = torch.tensor(np.linalg.inv(self._R), dtype = torch.double, device = self._device)


        def criterion_fn(hot_start):
            """
            Calculates the parts of the expected log-likelihood that are required for maximizing the LL w.r.t. the measurement
            parameters. That is, only \( Q_4 \) is calculated.

            Note that the sign of the LL is already flipped, such that the result of this function has to be minimized!
            """

            g_hat, G, hot_start = self._estimate_g_hat_and_G(hot_start)
            negative_log_likelihood = - torch.einsum('nit,nti,ii->', y, g_hat, R_inv) \
                                      - torch.einsum('nti,nit,ii->', g_hat, y, R_inv) \
                                      + torch.einsum('ntii,ii->', G, R_inv)
            return negative_log_likelihood, hot_start


        init_criterion, hot_start = criterion_fn(None)
        if self._do_lgds:
            self._optimize_g_linearly()
            return init_criterion.item(), 1, [init_criterion.item()]
        return self._optimize_g_sgd(lambda: criterion_fn(hot_start)[0])


    def _optimize_g_linearly(self):
        YX = np.einsum('nit,njt->ij', self._y, self._m_hat)
        self_correlation_sum = self._self_correlation.mean(axis = 0).sum(axis = 2)
        C_new = np.linalg.solve(self_correlation_sum.T, YX.T).T / self._no_sequences
        self._g_model.weight = torch.nn.Parameter(torch.tensor(C_new, dtype = torch.double), requires_grad = True)


    def _optimize_g_sgd(self, criterion_fn) -> Tuple[float, int, List[float]]:
        """
        Executed the actual gradient descent optimization of g.

        :param criterion_fn: The criterion function.
        :return: See _optimize_g return value.
        """

        optimizer = self._optimizer_factory()

        epsilon = torch.tensor(self._options.g_optimization_precision, device = self._device)
        criterion, criterion_prev = None, None
        iteration = 1
        history = []
        likelihood_observable = lambda: None if criterion is None else -criterion.item()
        if self._options.g_optimization_max_iterations is None:
            bar_widgets = ['G-Optimization:  ', NumberTrendWidget(EM.LIKELIHOOD_FORMAT, likelihood_observable)]
            bar_max_val = progressbar.widgets.UnknownLength
        else:
            bar_widgets = ['G-Optimization:  ', Percentage(), ' ', Bar(), ' ', ETA(), ' ', NumberTrendWidget(EM.LIKELIHOOD_FORMAT, likelihood_observable)]
            bar_max_val = self._options.g_optimization_max_iterations
        bar = progressbar.ProgressBar(widgets = bar_widgets, maxval = bar_max_val).start()
        while True:
            criterion = criterion_fn()
            history.append(-criterion.item())
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            bar.update(iteration)

            if criterion_prev is not None and (criterion - criterion_prev).abs() < epsilon:
                break
            if self._options.g_optimization_max_iterations is not None and iteration >= self._options.g_optimization_max_iterations:
                break

            criterion_prev = criterion
            iteration += 1
        bar.finish()

        return -criterion.item(), iteration, history


    def _estimate_g_hat_and_G(self, hot_start: Optional[Tuple[torch.tensor, torch.tensor, torch.tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.tensor, torch.tensor, torch.Tensor]]:
        """
        Estimates \( \hat{\vec{g}} \) and \( \mat{G} \) in one go using the batch processing of the cubature rule implementation.

        :param hot_start: If ``m_hat`` and ``V_hat`` have not changed, the result of a previous call to this function (the ``hot_start`` return value) can be passed here to not
                          copy the tensor data to the GPU of whatever device is used again.
        :return: ``tuple(g_hat, G, hot_start)``, where ``g_hat`` has the shape ``(N, T, p)`` and ``G`` has the shape ``(N, T, p, p)``.
        """

        if hot_start is None:
            m_hat = torch.tensor(self._m_hat, dtype = torch.double, device = self._device)
            V_hat = torch.tensor(self._V_hat, dtype = torch.double, device = self._device)

            m_hat_batch = m_hat.transpose(1, 2).reshape(-1, self._latent_dim)
            V_hat_batch = torch.einsum('bijt->btij', V_hat).reshape(-1, self._latent_dim, self._latent_dim)
            cov = V_hat_batch
        else:
            m_hat_batch, V_hat_batch, cov = hot_start

        g_hat_batch, _, _, cov = cubature.spherical_radial_torch(self._latent_dim, lambda x: self._g(x), m_hat_batch, cov, hot_start is not None)
        G_batch = cubature.spherical_radial_torch(self._latent_dim, lambda x: outer_batch_torch(self._g(x)), m_hat_batch, cov, True)[0]

        g_hat = g_hat_batch.view((self._no_sequences, self._T, self._observation_dim))
        G = G_batch.view((self._no_sequences, self._T, self._observation_dim, self._observation_dim))
        return g_hat, G, (m_hat_batch, V_hat_batch, cov)


    def _g(self, x: torch.Tensor) -> torch.Tensor:
        return self._g_model(x)


    def _g_numpy(self, x: np.ndarray) -> np.ndarray:
        return self._g(torch.tensor(x, device = self._device)).detach().cpu().numpy()


    def _calculate_likelihood(self) -> Optional[float]:
        if self._Q_problem or self._R_problem or self._V0_problem:
            return None

        # Store some variables to make the code below more readable.
        N = self._no_sequences
        p = self._observation_dim
        k = self._latent_dim
        T = self._T
        A = self._A
        B = self._B
        Q = self._Q
        Q_inv = np.linalg.inv(Q)
        m0 = self._m0.flatten()
        V0 = self._V0
        y = self._y
        u = self._u
        m_hat = self._m_hat
        R = self._R
        R_inv = np.linalg.inv(R)

        q1 = - N * T * (k + p) * np.log(2.0 * np.pi) \
             - N * np.log(np.linalg.det(V0)) \
             - N * (T - 1) * np.log(np.linalg.det(Q)) \
             - N * T * np.log(np.linalg.det(R))

        V0_inverse = np.linalg.inv(V0)
        q2_entry = lambda n: (m_hat[n, :, 0] - m0).T @ (V0_inverse @ (m_hat[n, :, 0] - m0))
        q2 = -np.sum([q2_entry(n) for n in range(N)], axis = 0)

        if self._do_control:
            q3_entry = lambda n, t: (m_hat[n, :, t] - A @ m_hat[n, :, t - 1] - B @ u[n, :, t - 1]).T @ (Q_inv @ (m_hat[n, :, t] - A @ m_hat[n, :, t - 1] - B @ u[n, :, t - 1]))
        else:
            q3_entry = lambda n, t: (m_hat[n, :, t] - A @ m_hat[n, :, t - 1]).T @ (Q_inv @ (m_hat[n, :, t] - A @ m_hat[n, :, t - 1]))
        q3 = -np.sum([q3_entry(n, t) for t in range(1, T) for n in range(N)], axis = 0)

        q4_entry = lambda n, t: (y[n, :, t] - self._g_numpy(m_hat[n, :, t])).T @ (R_inv @ (y[n, :, t] - self._g_numpy(m_hat[n, :, t])))
        q4 = -np.sum([q4_entry(n, t) for t in range(0, T) for n in range(N)], axis = 0)

        return (q1 + q2 + q3 + q4) / 2.0


    def _log(self, message):
        self._options.log(message)


    def get_estimations(self) -> Tuple[np.ndarray, Optional[np.ndarray], collections.OrderedDict, np.ndarray]:
        # If not doing the doubled to-call, CUDA gets an illegal memory access when moving something to the GPU next time.
        g_params = self._g_model.to('cpu').state_dict()
        self._g_model.to(self._device)
        return self._A, self._B, g_params, self._m0.reshape((-1,))


    def get_covariances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the estimated covariances.

        :return: (state_noise_cov, measurement_noise_cov, initial_state_cov, smoothed_state_covs)
            - state_noise_cov, shape (k, k): The state dynamics noise covariance.
            - measurement_noise_cov, shape (p, p): The measurement noise covariance.
            - initial_state_cov, shape (k, k): The initial state covariance/confidence.
            - smoothed_state_covs, shape (k, k, T): The covariances of the smoothed states, i.e. \( \Cov[s_{t - 1} | y_{1:T}] \).
        """
        return self._Q, self._R, self._V0, self._V_hat


    def get_estimated_latents(self) -> np.ndarray:
        return self._m_hat


    def get_problems(self) -> Tuple[bool, bool, bool]:
        return self._Q_problem, self._R_problem, self._V0_problem
