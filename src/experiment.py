import collections
import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    # Do regular LGDS instead of nonlinear measurements?
    do_lgds = False

    # Convergence checking configuration.
    epsilon = 0.00001
    max_iterations = 100
    g_optimization_learning_rate = 0.01
    g_optimization_precision = 1e-3
    g_optimization_max_iterations = 100

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
    dynamics_control_inputs_dim = 0

    # Observation model configuration.
    observation_model = None

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'  # Can be 'ode', 'image', 'manual' or 'gym'.
    dynamics_ode = None
    dynamics_params = { }
    dynamics_control_inputs = None
    initial_value_mean = None
    initial_value_cov = None
    dynamics_transform = None
    observation_cov = 0.0
    # Alternatively, the observations can be provided directly.
    dynamics_obs = None
    dynamics_manual_control_inputs = None
    # Alternatively, the observations can be generated from a gym environment.
    gym_do_control = True
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
    title = 'Pendulum (Gym), Control'

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    t_final = 20.0
    T = int(t_final / h)
    T_train = int(T / 2)
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
def pendulum_gym_no_control():
    # General experiment description.
    title = 'Pendulum (Gym), no Control'

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    t_final = 20.0
    T = int(t_final / h)
    T_train = int(T / 2)
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
    gym_do_control = False
    gym_environment = 'Pendulum-v0'
    gym_neutral_action = np.array([0.0])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def cartpole_gym():
    # General experiment description.
    title = 'Cartpole (Gym), Control'

    # Sequence configuration (time span and no. of sequences).
    h = 0.02
    t_final = 20.0
    T = int(t_final / h)
    T_train = T
    N = 1

    # Dimensionality configuration.
    latent_dim = 10
    observation_dim = 4
    observation_dim_names = ['Position (x)', 'Velocity (x)', 'Displacement', 'Velocity (Displacement)']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'CartPole-v1'



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def polynomial():
    # General experiment description.
    title = 'Polynomial Koopman'

    # Convergence checking configuration.
    g_optimization_max_iterations = 10000

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
    dynamics_ode = ['mu * x1', 'L * (x2 - x1 ** 2)']
    dynamics_params = { 'mu': -0.05, 'L': -1.0 }
    initial_value_mean = np.array([0.3, 0.4])
    initial_value_cov = np.diag([0.1, 0.1])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds():
    # General experiment description.
    title = 'Simple LGDS'

    # Do regular LGDS instead of nonlinear measurements?
    do_lgds = True

    # Convergence checking configuration.
    max_iterations = 100

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 24.0
    T = int(t_final / h)
    T_train = int(T / 8)
    N = 1

    # Dimensionality configuration.
    latent_dim = 2
    observation_dim = 2
    observation_dim_names = ['Dim. 1', 'Dim. 2']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', '-x1']
    dynamics_params = { }
    initial_value_mean = np.array([0.1, 0.2])
    initial_value_cov = np.diag([1e-5, 1e-5])



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds_simple_control():
    # General experiment description.
    title = 'Simple LGDS with Control'
    # Do regular LGDS instead of nonlinear measurements?
    do_lgds = True

    # Convergence checking configuration.
    max_iterations = 200

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T / h
    N = 1

    # Dimensionality configuration.
    latent_dim = 5
    observation_dim = 5
    observation_dim_names = list(['Dim. ' + str(dim) for dim in range(1, observation_dim + 1)])
    dynamics_control_inputs_dim = observation_dim

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'
    # Dynamics sampling configuration.
    dynamics_ode = list(['alpha * x%d + u%d' % (i, i) for i in range(1, observation_dim + 1)])
    dynamics_params = { 'alpha': 0.01 }
    dynamics_control_inputs = np.concatenate([np.random.uniform(-0.5, 0.5, size = (N, T_train, dynamics_control_inputs_dim)),
                                              np.zeros((N, T - T_train, dynamics_control_inputs_dim))], axis = 1)
    initial_value_mean = np.arange(observation_dim, dtype = np.float) + 1
    initial_value_cov = np.diag(np.zeros(observation_dim))



# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds_more_complicated_control():
    # General experiment description.
    title = 'More Complicated LGDS with Control'
    # Do regular LGDS instead of nonlinear measurements?
    do_lgds = True

    # Convergence checking configuration.
    max_iterations = 200

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T / h
    N = 2

    # Dimensionality configuration.
    latent_dim = 2
    observation_dim = 2
    observation_dim_names = ['Dim. 1', 'Dim. 2']
    dynamics_control_inputs_dim = observation_dim

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'
    # Dynamics sampling configuration.
    dynamics_ode = ['alpha * x1 + b + u1', 'alpha * x2 + u2']
    dynamics_params = { 'alpha': 0.01, 'b': -0.02 }
    dynamics_control_inputs = np.concatenate([np.concatenate([np.zeros((1, T_train, dynamics_control_inputs_dim)),
                                                              np.random.uniform(-1.0, 1.0, size = (N - 1, T_train, dynamics_control_inputs_dim))], axis = 0),
                                              np.zeros((N, T - T_train, dynamics_control_inputs_dim))], axis = 1)
    initial_value_mean = np.arange(observation_dim, dtype = np.float) + 1
    initial_value_cov = np.diag(np.zeros(observation_dim))



@ex.capture
def sample_ode(h: float, t_final: float, N: int, observation_dim: int, dynamics_control_inputs_dim: int, dynamics_ode: List[str], dynamics_params: Dict[str, float],
               dynamics_control_inputs: Union[Callable[[int, float, List[np.ndarray]], np.ndarray], List[List[np.ndarray]], np.ndarray],
               initial_value_mean: np.ndarray, initial_value_cov: np.ndarray, dynamics_transform: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert dynamics_ode is not None, 'dynamics_ode is not given!'
    assert dynamics_params is not None, 'dynamics_params is not given!'
    assert initial_value_mean is not None, 'initial_value_mean is not given!'
    assert observation_dim == len(dynamics_ode), 'observation_dim and dynamics_ode are inconsistent! Length of ODE must equal dimensionality.'
    assert observation_dim == initial_value_mean.shape[0], 'observation_dim and initial_value_mean are inconsistent! Length of initial value must equal dimensionality.'
    assert np.allclose(initial_value_cov, initial_value_cov.T), 'initial_value_cov is not symmetric!'
    assert (np.linalg.eigvals(initial_value_cov) >= 0).all(), 'initial_value_cov is not positive semi-definite!'
    assert observation_dim == initial_value_cov.shape[0], 'observation_dim and initial_value_cov are inconsistent! Size of initial value covariance must equal dimensionality.'


    def control_law(n: int, i: int, t: float, x: np.ndarray) -> np.ndarray:
        if dynamics_control_inputs is None:
            return np.array([])
        if callable(dynamics_control_inputs):
            return dynamics_control_inputs(n, t, x)
        if type(dynamics_control_inputs) == list or type(dynamics_control_inputs) == np.ndarray:
            return dynamics_control_inputs[n][i]
        raise Exception('Data type of control inputs not understood: %s' % str(type(dynamics_control_inputs)))


    sp_observation_params = ' '.join(['x%d' % i for i in range(1, observation_dim + 1)])
    sp_control_inputs_params = ' '.join(['u%d' % i for i in range(1, dynamics_control_inputs_dim + 1)])
    sp_params = sp.symbols('t %s %s' % (sp_observation_params, sp_control_inputs_params))
    ode_expr = [sp.lambdify(sp_params, sp.sympify(ode).subs(dynamics_params), 'numpy') for ode in dynamics_ode]
    transform_expr = None if dynamics_transform is None else [sp.lambdify(sp_params, sp.sympify(trans).subs(dynamics_params)) for trans in dynamics_transform]
    ode = lambda t, x, u: np.asarray([expr(t, *x, *u) for expr in ode_expr])
    sequences = []
    sequences_actions = []
    # noinspection PyUnresolvedReferences
    for n in range(0, N):
        if dynamics_control_inputs is None:
            # If we don't have control inputs, we can use more sophisticated integration methods.
            initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
            solution = sci.solve_ivp(lambda t, x: ode(t, x, []), (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau')
            # noinspection PyUnresolvedReferences
            t, trajectory = solution.t, solution.y.T
        else:
            t = np.arange(0.0, t_final, h)
            trajectory = []
            actions = []
            for i, tau in enumerate(t):
                if i == 0:
                    trajectory.append(np.random.multivariate_normal(initial_value_mean, initial_value_cov))
                else:
                    x = trajectory[-1]
                    action = control_law(n, i, tau, x)
                    trajectory.append(x + h * ode(tau, x, action))
                    actions.append(action)
            trajectory = np.asarray(trajectory)
            sequences_actions.append(actions)
        if transform_expr is None:
            sequences.append(trajectory)
        else:
            sequences.append(np.asarray([expr(t, *trajectory.T) for expr in transform_expr]).T)
    return np.asarray(sequences), None if dynamics_control_inputs is None else np.asarray(sequences_actions)



@ex.capture
def sample_gym(h: float, T: int, T_train: int, N: int, gym_do_control: bool, gym_environment: str, gym_neutral_action: np.ndarray, seed: int) \
        -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert gym_do_control is not None, 'gym_do_control is not given!'
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
            if gym_do_control and t < T_train:
                action = env.action_space.sample()
            else:
                action = gym_neutral_action
            sequence.append(env.step(action)[0].flatten())
            sequence_actions.append(np.asarray([action]).flatten())

        sequences.append(sequence)
        sequences_actions.append(sequence_actions)
    return np.asarray(sequences), np.asarray(sequences_actions) if gym_do_control else None



@ex.capture
def load_observations(dynamics_mode: str, h: float, t_final: float, T: int, T_train: int, dynamics_obs: np.ndarray,
                      dynamics_manual_control_inputs: np.ndarray, observation_cov: float) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    assert np.isclose(T * h, t_final), 'h, t_final and T are inconsistent! Result of T * h must equal t_final.'
    assert T_train <= T, 'T_train must be less or equal to T!'
    assert dynamics_mode is not None, 'dynamics_mode is not given!'
    assert dynamics_mode in ('ode', 'image', 'manual', 'gym'), 'dynamics_mode is not one of "ode", "image" or "gym"!'
    assert observation_cov is not None, 'observation_cov is not given!'
    assert observation_cov >= 0, 'observation_cov must be semi-positive!'

    if dynamics_mode == 'ode':
        sequences, sequences_actions = sample_ode()
    elif dynamics_mode == 'image':
        # dynamics_obs = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        # dynamics_obs_noisy = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d_noisy-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        raise Exception('Image dynamics_mode is currently not supported!')
    elif dynamics_mode == 'manual':
        sequences = dynamics_obs
        sequences_actions = dynamics_manual_control_inputs
    elif dynamics_mode == 'gym':
        sequences, sequences_actions = sample_gym()
    else:
        assert False, 'Should never happen.'
    sequences_noisy = sequences + np.random.multivariate_normal(np.array([0.0]), np.array([[observation_cov]]), size = sequences.shape).reshape(sequences.shape)
    return sequences, sequences_noisy, sequences_actions



@ex.capture
def load_observation_model(do_lgds: bool, latent_dim: int, observation_dim: int, observation_model: Union[str, List[str]]):
    if do_lgds:
        model = torch.nn.Linear(latent_dim, observation_dim, bias = False)
        torch.nn.init.eye_(model.weight)
    else:
        model = util.build_dynamic_model(observation_model, latent_dim, observation_dim)
    return model



def build_result_dict(iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, control_inputs: Optional[np.ndarray], latents: np.ndarray, A: np.ndarray,
                      B: Optional[np.ndarray], g_params: collections.OrderedDict, m0: np.ndarray, Q: np.ndarray, R: np.ndarray, V0: np.ndarray, V_hat: np.ndarray,
                      log_likelihood: Optional[float]):
    result_dict = {
            'iterations':     iterations,
            'log_likelihood': log_likelihood,
            'input':          {
                    'observations':       observations.copy(),
                    'observations_noisy': observations_noisy.copy(),
                    'control_inputs':     None if control_inputs is None else control_inputs.copy()
            },
            'estimations':    {
                    'latents':  latents.copy(),
                    'A':        A.copy(),
                    'B':        None if B is None else B.copy(),
                    'g_params': g_params.copy(),
                    'm0':       m0.copy(),
                    'Q':        Q.copy(),
                    'R':        R.copy(),
                    'V0':       V0.copy(),
                    'V_hat':    V_hat.copy()
            }
    }
    return result_dict



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, do_lgds, title, epsilon, max_iterations, g_optimization_learning_rate, g_optimization_precision, g_optimization_max_iterations,
         create_checkpoint_every_n_iterations, load_initialization_from_file, T_train, latent_dim):
    if title is None:
        raise ExperimentNotConfiguredInterrupt()

    observations_all, observations_all_noisy, control_inputs = load_observations()
    observations_train_noisy = observations_all_noisy[:, :T_train, :]
    control_inputs_train = None if control_inputs is None else control_inputs[:, :T_train - 1, :]  # The last state does not have an action.


    def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)
        _run.log_scalar('g_ll', g_ll, iteration)
        _run.log_scalar('g_iterations', g_iterations, iteration)
        for i, ll in enumerate(g_ll_history):
            _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

        if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
            A_cp, B_cp, g_params_cp, m0_cp = em.get_estimations()
            Q_cp, R_cp, V0_cp, V_hat_cp = em.get_covariances()
            checkpoint = build_result_dict(iteration, observations_all, observations_all_noisy, control_inputs, em.get_estimated_latents(), A_cp, B_cp, g_params_cp, m0_cp,
                                           Q_cp, R_cp, V0_cp, V_hat_cp, None)
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

    g = load_observation_model()

    options = EMOptions()
    options.do_lgds = do_lgds
    options.precision = epsilon
    options.max_iterations = max_iterations
    options.log = _log.info
    options.g_optimization_learning_rate = g_optimization_learning_rate
    options.g_optimization_precision = g_optimization_precision
    options.g_optimization_max_iterations = g_optimization_max_iterations
    em = EM(latent_dim, observations_train_noisy, control_inputs_train, model = g, initialization = initialization, options = options)
    log_likelihoods = em.fit(callback = callback)
    A_est, B_est, g_params_est, m0_est = em.get_estimations()
    Q_est, R_est, V0_est, V_hat_est = em.get_covariances()
    latents = em.get_estimated_latents()

    Q_problem, R_problem, V0_problem = em.get_problems()
    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    return build_result_dict(len(log_likelihoods), observations_all, observations_all_noisy, control_inputs, latents, A_est, B_est, g_params_est, m0_est, Q_est, R_est, V0_est,
                             V_hat_est, log_likelihoods[-1])
