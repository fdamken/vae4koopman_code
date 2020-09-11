import collections
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import gym
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


util.apply_sacred_frame_error_workaround()

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
    g_optimization_max_iterations = 10000

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
    dynamics_mode = 'ode'  # Can be 'ode', 'image' or 'gym'.
    dynamics_ode = None
    dynamics_params = { }
    initial_value_mean = None
    initial_value_cov = None
    dynamics_transform = None
    observation_cov = 0.0
    # Alternatively, the observations can be provided directly.
    dynamics_obs = None
    dynamics_obs_noisy = None
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = None
    gym_neutral_action = None



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
    observation_dim_names = ['Position', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

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
    latent_dim = 10
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
def pendulum_gym():
    # General experiment description.
    title = 'Damped Pendulum'

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    T = 200
    T_train = 150
    t_final = T * h
    N = 1

    # Dimensionality configuration.
    latent_dim = 10
    observation_dim = 3
    observation_dim_names = ['Position (x)', 'Position (y)', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'Pendulum-v0'
    gym_neutral_action = np.array([0.0])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_damped_xy():
    # General experiment description.
    title = 'Damped Pendulum (From xy Coordinates)'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 10
    observation_dim = 2
    observation_dim_names = ['x', 'y']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1) - d * x2']
    dynamics_params = { 'd': 0.1 }
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])
    dynamics_transform = ['sin(x1)', '-cos(x1)']



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_damped_from_images():
    # General experiment description.
    title = 'Damped Pendulum (Image-Based)'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 10
    observation_dim = 16 * 16
    observation_dim_names = list(['Dim. %3d' % i for i in range(observation_dim)])

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, 100)', 'Tanh()', 'Linear(100, 200)', 'Tanh()', 'Linear(200, out_features)', 'Tanh()']

    # Dynamics sampling configuration.
    dynamics_mode = 'image'
    # Observations.
    dynamics_obs = True
    dynamics_obs_noisy = True



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
def sample_dynamics(h: float, t_final: float, N: int, observation_dim: int, dynamics_ode: List[str], dynamics_params: Dict[str, float], initial_value_mean: np.ndarray,
                    initial_value_cov: np.ndarray, dynamics_transform: List[str]) -> np.ndarray:
    assert dynamics_ode is not None, 'dynamics_ode is not given!'
    assert dynamics_params is not None, 'dynamics_params is not given!'
    assert initial_value_mean is not None, 'initial_value_mean is not given!'
    assert observation_dim == len(dynamics_ode), 'observation_dim and dynamics_ode are inconsistent! Length of ODE must equal dimensionality.'
    assert observation_dim == initial_value_mean.shape[0], 'observation_dim and initial_value_mean are inconsistent! Length of initial value must equal dimensionality.'
    assert np.allclose(initial_value_cov, initial_value_cov.T), 'initial_value_cov is not symmetric!'
    assert (np.linalg.eigvals(initial_value_cov) >= 0).all(), 'initial_value_cov is not positive semi-definite!'
    assert observation_dim == initial_value_cov.shape[0], 'observation_dim and initial_value_cov are inconsistent! Size of initial value covariance must equal dimensionality.'

    sp_params = sp.symbols('t ' + ' '.join(['x%d' % i for i in range(1, observation_dim + 1)]))
    ode_expr = [sp.lambdify(sp_params, sp.sympify(ode).subs(dynamics_params), 'numpy') for ode in dynamics_ode]
    transform_expr = None if dynamics_transform is None else [sp.lambdify(sp_params, sp.sympify(trans).subs(dynamics_params)) for trans in dynamics_transform]
    ode = lambda t, x: np.asarray([expr(t, *x) for expr in ode_expr])
    sequences = []
    # noinspection PyUnresolvedReferences
    for _ in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        solution = sci.solve_ivp(ode, (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau')
        # noinspection PyUnresolvedReferences
        t, trajectory = solution.t, solution.y
        if transform_expr is None:
            sequences.append(trajectory.T)
        else:
            sequences.append(np.asarray([expr(t, *trajectory) for expr in transform_expr]).T)
    return np.asarray(sequences)



@ex.capture
def sample_gym(h: float, T: int, T_train: int, N: int, gym_environment: str, gym_neutral_action: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    assert gym_environment is not None, 'gym_environment is not given!'
    assert T == T_train or gym_neutral_action is not None, 'gym_neutral_action is not given, but test data exists!'

    env = gym.make(gym_environment)
    env.seed(seed)
    env.dt = h
    sequences = []
    sequences_actions = []
    for n in range(N):
        sequence = []
        sequence_actions = []

        sequence.append(env.reset())
        for t in range(1, T):
            action = env.action_space.sample() if t < T_train else gym_neutral_action
            sequence.append(env.step(action)[0])
            sequence_actions.append(action)

        sequences.append(sequence)
        sequences_actions.append(sequence_actions)
    return np.asarray(sequences), np.asarray(sequences_actions)



@ex.capture
def load_observations(dynamics_mode: str, h: float, t_final: float, T: int, T_train: int, observation_cov: float) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    assert np.isclose(T * h, t_final), 'h, t_final and T are inconsistent! Result of T * h must equal t_final.'
    assert T_train <= T, 'T_train must be less or equal to T!'
    assert dynamics_mode is not None, 'dynamics_mode is not given!'
    assert dynamics_mode in ('ode', 'image', 'gym'), 'dynamics_mode is not one of "ode", "image" or "gym"!'
    assert observation_cov is not None, 'observation_cov is not given!'
    assert observation_cov >= 0, 'observation_cov must be semi-positive!'

    if dynamics_mode == 'ode':
        sequences, sequences_actions = sample_dynamics(), None
    elif dynamics_mode == 'image':
        # dynamics_obs = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        # dynamics_obs_noisy = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d_noisy-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        raise Exception('Image dynamics_mode is currently not supported!')
    elif dynamics_mode == 'gym':
        sequences, sequences_actions = sample_gym()
    else:
        assert False, 'Should never happen.'
    sequences_noisy = sequences + np.random.multivariate_normal(np.array([0.0]), np.array([[observation_cov]]), size = sequences.shape).reshape(sequences.shape)
    return sequences, sequences_noisy, sequences_actions



def build_result_dict(iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, control_inputs: Optional[np.ndarray], latents: np.ndarray,
                      A: np.ndarray, B: Optional[np.ndarray], Q: np.ndarray, g_params: collections.OrderedDict, R: np.ndarray, m0: np.ndarray, V0: np.ndarray,
                      log_likelihood: Optional[float]):
    result_dict = {
            'iterations':  iterations,
            'input':       {
                    'observations':       observations.copy(),
                    'observations_noisy': observations_noisy.copy(),
                    'control_inputs':     control_inputs.copy()
            },
            'estimations': {
                    'latents':  latents.copy(),
                    'A':        A.copy(),
                    'B':        B.copy(),
                    'Q':        Q.copy(),
                    'g_params': g_params.copy(),
                    'R':        R.copy(),
                    'm0':       m0.copy(),
                    'V0':       V0.copy()
            }
    }
    if log_likelihood is not None:
        result_dict['log_likelihood'] = log_likelihood
    return result_dict



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, title, epsilon, max_iterations, g_optimization_learning_rate, g_optimization_precision, g_optimization_max_iterations,
         create_checkpoint_every_n_iterations, load_initialization_from_file, T_train, latent_dim, observation_dim, observation_model):
    if title is None:
        raise ExperimentNotConfiguredInterrupt()

    observations_all, observations_all_noisy, control_inputs = load_observations()
    observations_train_noisy = observations_all_noisy[:, :T_train, :]


    def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)
        _run.log_scalar('g_ll', g_ll, iteration)
        _run.log_scalar('g_iterations', g_iterations, iteration)
        for i, ll in enumerate(g_ll_history):
            _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

        if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
            A_cp, B_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp = em.get_estimations()
            checkpoint = build_result_dict(iteration, observations_all, observations_all_noisy, control_inputs, em.get_estimated_latents(),
                                           A_cp, B_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp, None)
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
    em = EM(latent_dim, observations_train_noisy, control_inputs, model = g, initialization = initialization, options = options)
    log_likelihoods = em.fit(callback = callback)
    A_est, B_est, Q_est, g_params_est, R_est, m0_est, V0_est = em.get_estimations()
    latents = em.get_estimated_latents()

    Q_problem, R_problem, V0_problem = em.get_problems()
    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    return build_result_dict(len(log_likelihoods), observations_all, observations_all_noisy, control_inputs, latents,
                             A_est, B_est, Q_est, g_params_est, R_est, m0_est, V0_est, log_likelihoods[-1])
