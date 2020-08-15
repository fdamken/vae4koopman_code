import collections
import os
import tempfile
from typing import Dict, List, Optional

import jsonpickle
import numpy as np
import scipy.integrate as sci
import sympy as sp
import torch
from neptunecontrib.monitoring.sacred import NeptuneObserver
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src import util
from src.em import EM, EMInitialization, EMOptions
from src.util import ExperimentNotConfiguredInterrupt, MatrixProblemInterrupt


torch.set_default_dtype(torch.double)

ex = Experiment('generated-observations')
ex.observers.append(FileStorageObserver('tmp_results'))
if os.environ.get('NO_NEPTUNE') is None:
    ex.observers.append(NeptuneObserver(project_name = 'fdamken/variational-koopman'))



# noinspection PyUnusedLocal,PyPep8Naming
@ex.config
def defaults():
    # General experiment description.
    title = None
    seed = 42
    create_checkpoint_every_n_iterations = 5
    load_initialization_from_file = None

    # Convergence checking configuration.
    epsilon = 0.00001
    max_iterations = 100
    g_optimization_learning_rate = 0.01
    g_optimization_precision = 1e-3
    g_optimization_max_iterations = 100
    log_g_optimization_progress = True

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = None
    observation_dim = None
    observation_dim_names = []

    # Observation model configuration.
    observation_model = None

    # Dynamics sampling configuration.
    dynamics_ode = None
    dynamics_params = { }
    initial_value_mean = None
    initial_value_cov = None
    observation_cov = 0.0



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum():
    # General experiment description.
    title = 'Pendulum'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 3
    observation_dim = 2
    observation_dim_names = ['Dim. 1', 'Dim. 2']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50_features)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1)']
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_damped():
    # General experiment description.
    title = 'Damped Pendulum'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 3
    observation_dim = 2
    observation_dim_names = ['Position', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1) - d * x2']
    dynamics_params = { 'd': 0.1 }
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def polynomial():
    # General experiment description.
    title = 'Polynomial Koopman'

    # Convergence checking configuration.
    g_optimization_max_iterations = None

    # Sequence configuration (time span and no. of sequences).
    h = 0.02
    t_final = 2 * 1.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 3
    observation_dim = 2
    observation_dim_names = ['Position', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 10)', 'Tanh()', 'Linear(10, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['mu * x1', 'lambda * (x2 - x1 ** 2)']
    dynamics_params = { 'mu': -0.05, 'lambda': -1.0 }
    initial_value_mean = np.array([0.3, 0.4])
    initial_value_cov = np.diag([0.1, 0.1])



@ex.capture
def sample_dynamics(h: float, t_final: float, T: int, N: int, observation_dim: int, dynamics_ode: List[str], dynamics_params: Dict[str, float], initial_value_mean: np.ndarray,
                    initial_value_cov: np.ndarray, observation_cov: float):
    assert np.isclose(T * h, t_final), 'h, t_final and T are inconsistent! Result of T * h must equal t_final.'
    assert observation_dim == len(dynamics_ode), 'observation_dim and dynamics_ode are inconsistent! Length of ODE must equal dimensionality.'
    assert observation_dim == initial_value_mean.shape[0], 'observation_dim and initial_value_mean are inconsistent! Length of initial value must equal dimensionality.'
    assert np.allclose(initial_value_cov, initial_value_cov.T), 'initial_value_cov is not symmetric!'
    assert (np.linalg.eigvals(initial_value_cov) >= 0).all(), 'initial_value_cov is not positive semi-definite!'
    assert observation_dim == initial_value_cov.shape[0], 'observation_dim and initial_value_cov are inconsistent! Size of initial value covariance must equal dimensionality.'
    assert observation_cov >= 0, 'observation_cov must be semi-positive!'

    sp_params = sp.symbols('t ' + ' '.join(['x%d' % i for i in range(1, observation_dim + 1)]))
    ode_expr = [sp.lambdify(sp_params, sp.sympify(ode).subs(dynamics_params), 'numpy') for ode in dynamics_ode]
    ode = lambda t, x: np.asarray([expr(t, *x) for expr in ode_expr])

    sequences = []
    for _ in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        sequences.append(sci.solve_ivp(ode, (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau').y.T)
    sequences = np.asarray(sequences)
    sequences_noisy = sequences + np.random.normal(loc = 0, scale = np.sqrt(observation_cov), size = sequences.shape)
    return sequences, sequences_noisy



def build_result_dict(iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, latents: np.ndarray, A: np.ndarray, Q: np.ndarray,
                      g_params: collections.OrderedDict, R: np.ndarray, m0: np.ndarray, V0: np.ndarray, log_likelihood: Optional[float]):
    result_dict = {
            'iterations':  iterations,
            'input':       {
                    'observations':       observations,
                    'observations_noisy': observations_noisy
            },
            'estimations': {
                    'latents':  latents,
                    'A':        A,
                    'Q':        Q,
                    'g_params': g_params,
                    'R':        R,
                    'm0':       m0,
                    'V0':       V0
            }
    }
    if log_likelihood is not None:
        result_dict['log_likelihood'] = log_likelihood
    return result_dict



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, title, epsilon, max_iterations, g_optimization_learning_rate, g_optimization_precision, g_optimization_max_iterations, log_g_optimization_progress,
         create_checkpoint_every_n_iterations, load_initialization_from_file, T_train, latent_dim, observation_dim, observation_model):
    if title is None:
        raise ExperimentNotConfiguredInterrupt()

    observations_all, observations_all_noisy = sample_dynamics()
    observations_train_noisy = observations_all_noisy[:, :T_train, :]


    def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)
        _run.log_scalar('g_ll', g_ll, iteration)
        _run.log_scalar('g_iterations', g_iterations, iteration)
        for i, ll in enumerate(g_ll_history):
            _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

        if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
            A_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp = em.get_estimations()
            checkpoint = build_result_dict(iteration, observations_all, observations_all_noisy, em.get_estimated_latents(), A_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp, None)
            _, f_path = tempfile.mkstemp(prefix = 'checkpoint_%05d-' % iteration, suffix = '.json')
            with open(f_path, 'w') as f:
                f.write(jsonpickle.dumps({ 'result': checkpoint }))
            _run.add_artifact(f_path, 'checkpoint_%05d.json' % iteration, metadata = { 'iteration': iteration })
            os.remove(f_path)


    initialization = EMInitialization()
    if load_initialization_from_file is not None:
        with open(load_initialization_from_file) as f:
            initialization = jsonpickle.loads(f.read())['result']['estimations']
        initialization.A = initialization['A']
        initialization.Q = initialization['Q']
        initialization.g = initialization['g_params']
        initialization.R = initialization['R']
        initialization.m0 = initialization['m0']
        initialization.V0 = initialization['V0']

    g = util.build_dynamic_model(observation_model, latent_dim, observation_dim)

    options = EMOptions()
    options.precision = epsilon
    options.max_iterations = max_iterations
    options.log = _log.info
    options.g_optimization_learning_rate = g_optimization_learning_rate
    options.g_optimization_precision = g_optimization_precision
    options.g_optimization_max_iterations = g_optimization_max_iterations
    options.log_g_optimization_progress = log_g_optimization_progress
    em = EM(latent_dim, observations_train_noisy, model = g, initialization = initialization, options = options)
    log_likelihoods = em.fit(callback = callback)
    A_est, Q_est, g_params_est, R_est, m0_est, V0_est = em.get_estimations()
    latents = em.get_estimated_latents()

    Q_problem, R_problem, V0_problem = em.get_problems()
    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    return build_result_dict(len(log_likelihoods), observations_all, observations_all_noisy, latents, A_est, Q_est, g_params_est, R_est, m0_est, V0_est, log_likelihoods[-1])
