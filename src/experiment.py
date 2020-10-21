import collections
import os
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import gym
import jsonpickle
import numpy as np
import progressbar
import scipy.integrate as sci
import sympy as sp
import torch
from neptunecontrib.monitoring.sacred import NeptuneObserver
from progressbar import Bar, ETA, Percentage
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
    ex.observers.append(NeptuneObserver(project_name='fdamken/variational-koopman'))


# noinspection PyUnusedLocal,PyPep8Naming
@ex.config
def defaults():
    # General experiment description.
    title = None
    seed = 42
    create_checkpoint_every_n_iterations = 5
    load_initialization_from_file = None
    estimate_diagonal_noise = False
    do_whitening = False
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
    dynamics_params = {}
    dynamics_control_inputs = None
    dynamics_neutral_control = None
    initial_value_mean = None
    initial_value_cov = None
    dynamics_transform = None
    observation_cov = 0.0
    # Alternatively, the observations can be provided directly.
    dynamics_obs = None
    dynamics_obs_without_actions = None
    dynamics_manual_control_inputs = None
    dynamics_manual_neutral_control = None
    # Alternatively, the observations can be generated from a gym environment.
    gym_do_control = True
    gym_environment = None
    gym_neutral_action = None
    gym_render = False


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
    latent_dim = 10
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
def pendulum_with_control():
    # General experiment description.
    title = 'Pendulum (Control)'

    # Convergence checking configuration.
    max_iterations = 200

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
    dynamics_control_inputs_dim = 1

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1) + u1']
    dynamics_control_inputs = 'Random.Uniform(0.1)'
    dynamics_neutral_control = np.array([0.0])
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([0.0, 0.0])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_damped():
    # General experiment description.
    title = 'Damped Pendulum'

    # Convergence checking configuration.
    max_iterations = 200

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
    dynamics_params = {'d': 0.1}
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_gym():
    # General experiment description.
    title = 'Pendulum (Gym), Control'

    # Convergence checking configuration.
    max_iterations = 500

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    t_final = 20.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 5

    # Dimensionality configuration.
    latent_dim = 4
    observation_dim = 3
    observation_dim_names = ['Position (x)', 'Position (y)', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'Pendulum-v0'
    gym_neutral_action = np.array([0.0])
    observation_cov = 1e-5


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_gym_like_morton():
    # General experiment description.
    title = 'Pendulum (Gym), Control, Like Morton'
    do_whitening = True

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    T_train = 16
    T = 2 * T_train
    t_final = T * h
    N = 4400

    # Dimensionality configuration.
    latent_dim = 4
    observation_dim = 3
    observation_dim_names = ['Position (x)', 'Position (y)', 'Velocity']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 64)', 'ELU()', 'Linear(64, 64)', 'ELU()', 'Linear(64, out_features)']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_do_control = False
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
def lunar_lander_gym():
    # General experiment description.
    title = 'Lunar Lander (Gym), Control'
    estimate_diagonal_noise = True
    max_iterations = 500

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T * h
    N = 10

    # Dimensionality configuration.
    latent_dim = 10
    observation_dim = 8
    observation_dim_names = [r'$x$', r'$y$', r'$\dot{x}$', r'$\dot{y}$', r'$\theta$', r'$\dot{\theta}}$']

    # Observation model configuration.
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    dynamics_transform = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'LunarLanderContinuous-v2'
    gym_neutral_action = np.array([0.0, 0.0])


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
def harmonic_oscillator():
    # General experiment description.
    title = 'Harmonic Oscillator'

    # Convergence checking configuration.
    max_iterations = 100

    # Sequence configuration (time span and no. of sequences).
    h = 0.01
    t_final = 2 * 10.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    latent_dim = 2
    observation_dim = 2
    observation_dim_names = ['Position', 'Velocity']
    # dynamics_control_inputs_dim = 1

    # Observation model configuration.
    observation_model = ['Linear(in_features, out_features)']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', '-x1']
    # dynamics_control_inputs = lambda n, t, x: -np.asarray([x[1]])
    # dynamics_control_inputs = 'Random.Uniform(0.1)'
    dynamics_neutral_control = np.array([0.0])
    initial_value_mean = np.array([1.0, 0.0])
    initial_value_cov = np.diag([0.0, 0.0])
    # observation_cov = 1e-5


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
    dynamics_params = {'mu': -0.05, 'L': -1.0}
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
    T_train = int(T / 2)
    N = 3

    # Dimensionality configuration.
    latent_dim = 5
    observation_dim = 2
    observation_dim_names = ['Dim. 1', 'Dim. 2']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', '-x1']
    dynamics_params = {}
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
    h = 0.5
    T = 200
    T_train = 150
    t_final = 100
    N = 3

    # Dimensionality configuration.
    latent_dim = 5
    observation_dim = 5
    observation_dim_names = list(['Dim. ' + str(dim) for dim in range(1, observation_dim + 1)])
    dynamics_control_inputs_dim = observation_dim

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'
    # Dynamics sampling configuration.
    dynamics_ode = list(['alpha * x%d + u%d' % (i, i) for i in range(1, observation_dim + 1)])
    dynamics_params = {'alpha': 0.01}
    dynamics_control_inputs = 'Random.Uniform(0.1)'
    dynamics_neutral_control = np.zeros(observation_dim)
    initial_value_mean = np.arange(observation_dim, dtype=np.float) + 1
    initial_value_cov = np.diag(np.zeros(observation_dim))


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds_more_complicated_control():
    # General experiment description.
    title = 'More Complicated LGDS with Control'
    # Do regular LGDS instead of nonlinear measurements?
    do_lgds = True

    # Convergence checking configuration.
    max_iterations = 150

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T / h
    N = 5

    # Dimensionality configuration.
    latent_dim = 2
    observation_dim = 2
    observation_dim_names = ['Dim. 1', 'Dim. 2']
    dynamics_control_inputs_dim = observation_dim

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'
    # Dynamics sampling configuration.
    dynamics_ode = ['alpha * x1 + b + u1', 'alpha * x2 + u2']
    dynamics_params = {'alpha': 0.01, 'b': -0.02}
    dynamics_control_inputs = 'Random.Uniform(1.0)'
    dynamics_neutral_control = np.zeros(observation_dim)
    initial_value_mean = np.arange(observation_dim, dtype=np.float) + 1
    initial_value_cov = np.diag(np.zeros(observation_dim))


@ex.capture
def sample_ode(h: float, t_final: float, T: int, T_train: int, N: int, observation_dim: int, dynamics_control_inputs_dim: int, dynamics_ode: List[str],
               dynamics_params: Dict[str, float], dynamics_control_inputs: Union[Callable[[int, float, List[np.ndarray]], np.ndarray], List[List[np.ndarray]], np.ndarray, str],
               dynamics_neutral_control: np.ndarray, initial_value_mean: np.ndarray, initial_value_cov: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert dynamics_ode is not None, 'dynamics_ode is not given!'
    assert dynamics_params is not None, 'dynamics_params is not given!'
    assert initial_value_mean is not None, 'initial_value_mean is not given!'
    assert observation_dim == len(dynamics_ode), 'observation_dim and dynamics_ode are inconsistent! Length of ODE must equal dimensionality.'
    assert observation_dim == initial_value_mean.shape[0], 'observation_dim and initial_value_mean are inconsistent! Length of initial value must equal dimensionality.'
    assert np.allclose(initial_value_cov, initial_value_cov.T), 'initial_value_cov is not symmetric!'
    assert (np.linalg.eigvals(initial_value_cov) >= 0).all(), 'initial_value_cov is not positive semi-definite!'
    assert observation_dim == initial_value_cov.shape[0], 'observation_dim and initial_value_cov are inconsistent! Size of initial value covariance must equal dimensionality.'

    if type(dynamics_control_inputs) == str and dynamics_control_inputs.startswith('Random.'):
        dynamics_control_inputs = np.concatenate([util.random_from_descriptor(dynamics_control_inputs, (N, T_train, dynamics_control_inputs_dim)),
                                                  np.zeros((N, T - T_train, dynamics_control_inputs_dim))], axis=1)

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
    ode = lambda t, x, u: np.asarray([expr(t, *x, *u) for expr in ode_expr])
    sequences = []
    sequences_without_actions = []
    sequences_actions = []
    # noinspection PyUnresolvedReferences
    for n in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        if dynamics_control_inputs is None:
            # If we don't have control inputs, we can use more sophisticated integration methods.
            solution = sci.solve_ivp(lambda t, x: ode(t, x, []), (0, t_final), initial_value, t_eval=np.arange(0, t_final, h), method='Radau')
            # noinspection PyUnresolvedReferences
            t, trajectory = solution.t, solution.y.T
            trajectory_without_actions = trajectory
        else:
            t = np.arange(0.0, t_final, h)
            trajectory = []
            trajectory_without_actions = []
            actions = []
            for i, tau in enumerate(t):
                if i == 0:
                    trajectory.append(initial_value)
                    trajectory_without_actions.append(initial_value)
                else:
                    x = trajectory[-1]
                    if i < T_train:
                        action = control_law(n, i, tau, x)
                    else:
                        action = dynamics_neutral_control
                    trajectory.append(x + h * ode(tau, x, action))
                    actions.append(action)

                    x_wo_control = trajectory_without_actions[-1]
                    trajectory_without_actions.append(x_wo_control + h * ode(tau, x_wo_control, dynamics_neutral_control))
            trajectory = np.asarray(trajectory)
            sequences_actions.append(actions)
        sequences.append(trajectory)
        sequences_without_actions.append(trajectory_without_actions)
    return np.asarray(sequences), np.asarray(sequences_without_actions), None if dynamics_control_inputs is None else np.asarray(sequences_actions), dynamics_neutral_control


@ex.capture
def sample_manual(dynamics_obs: np.ndarray, dynamics_obs_without_actions: np.ndarray, dynamics_manual_control_inputs: np.ndarray, dynamics_manual_neutral_control: np.ndarray):
    return dynamics_obs, dynamics_obs_without_actions, dynamics_manual_control_inputs, dynamics_manual_neutral_control


@ex.capture
def sample_gym(h: float, T: int, T_train: int, N: int, gym_do_control: bool, gym_environment: str, gym_neutral_action: np.ndarray, gym_render: bool, seed: int) \
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert gym_do_control is not None, 'gym_do_control is not given!'
    assert gym_environment is not None, 'gym_environment is not given!'
    assert T == T_train or gym_neutral_action is not None, 'gym_neutral_action is not given, but test data exists!'

    # Create controlled and uncontrolled environments.
    env = gym.make(gym_environment)
    env.action_space.seed(seed)
    env.dt = h
    env_without_control = gym.make(gym_environment)
    env_without_control.action_space.seed(seed)
    env_without_control.dt = h
    sequences = []
    sequences_without_control = []
    sequences_actions = []
    bar = progressbar.ProgressBar(widgets=['Sampling Gym: ', Percentage(), ' ', Bar(), ' ', ETA()], maxval=N * T).start()
    for n in range(N):
        sequence = []
        sequence_without_control = []
        sequence_actions = []

        env.seed(seed)
        env_without_control.seed(seed)
        initial_state = env.reset()
        initial_state_without_control = env_without_control.reset()
        sequence.append(initial_state)
        sequence_without_control.append(initial_state_without_control)
        for t in range(1, T):
            if gym_do_control and t < T_train:
                action = env.action_space.sample()
            else:
                action = gym_neutral_action
            state = env.step(action)[0].flatten()
            sequence.append(state)
            sequence_without_control.append(env_without_control.step(gym_neutral_action)[0].flatten())
            sequence_actions.append(np.asarray([action]).flatten())

            if gym_render:
                env.render()

            bar.update(n * T + t - 1)

        sequences.append(sequence)
        sequences_without_control.append(sequence_without_control)
        sequences_actions.append(sequence_actions)

        env.close()
    bar.finish()
    return np.asarray(sequences), np.asarray(sequences_without_control), np.asarray(sequences_actions) if gym_do_control else None, gym_neutral_action if gym_do_control else None


@ex.capture
def load_observations(dynamics_mode: str, h: float, t_final: float, T: int, T_train: int, observation_cov: float, observation_dim: int, dynamics_control_inputs_dim: int,
                      dynamics_params: Dict[str, float], dynamics_transform: List[str]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert np.isclose(T * h, t_final), 'h, t_final and T are inconsistent! Result of T * h must equal t_final.'
    assert T_train <= T, 'T_train must be less or equal to T!'
    assert dynamics_mode is not None, 'dynamics_mode is not given!'
    assert dynamics_mode in ('ode', 'image', 'manual', 'gym'), 'dynamics_mode is not one of "ode", "image" or "gym"!'
    assert observation_cov is not None, 'observation_cov is not given!'
    assert observation_cov >= 0, 'observation_cov must be semi-positive!'

    if dynamics_mode == 'ode':
        sequences, sequences_without_actions, sequences_actions, neutral_action = sample_ode()
    elif dynamics_mode == 'image':
        # dynamics_obs = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        # dynamics_obs_noisy = util.bw_image(
        #        np.asarray([[imageio.imread('data/tmp_pendulum/sequence-%05d_noisy-%06.3f.bmp' % (n, t)).flatten() for t in np.arange(0.0, t_final, h)] for n in range(N)]))
        raise Exception('Image dynamics_mode is currently not supported!')
    elif dynamics_mode == 'manual':
        sequences, sequences_without_actions, sequences_actions, neutral_action = sample_manual()
    elif dynamics_mode == 'gym':
        sequences, sequences_without_actions, sequences_actions, neutral_action = sample_gym()
    else:
        assert False, 'Should never happen.'
    if dynamics_transform is not None:
        sp_observation_params = ' '.join(['x%d' % i for i in range(1, observation_dim + 1)])
        sp_control_inputs_params = ' '.join(['u%d' % i for i in range(1, dynamics_control_inputs_dim + 1)])
        sp_params = sp.symbols('t %s %s' % (sp_observation_params, sp_control_inputs_params))
        transform_expr = None if dynamics_transform is None else [sp.lambdify(sp_params, sp.sympify(trans).subs(dynamics_params)) for trans in dynamics_transform]
        sequences_new = []
        sequences_without_actions_new = []
        t = np.arange(0.0, t_final, h)
        for trajectory, trajectory_without_actions in zip(sequences, sequences_without_actions):
            sequences_new.append(np.asarray([expr(t, *trajectory.T) for expr in transform_expr]).T)
            sequences_without_actions_new.append(np.asarray([expr(t, *trajectory_without_actions.T) for expr in transform_expr]).T)
        sequences = np.asarray(sequences_new)
        sequences_without_actions = np.asarray(sequences_without_actions_new)
    sequences_noisy = sequences + np.random.multivariate_normal(np.array([0.0]), np.array([[observation_cov]]), size=sequences.shape).reshape(sequences.shape)
    return sequences, sequences_noisy, sequences_without_actions, sequences_actions, neutral_action


@ex.capture
def load_observation_model(do_lgds: bool, latent_dim: int, observation_dim_names: List[str], observation_model: Union[str, List[str]]):
    observation_dim = len(observation_dim_names)
    linear = do_lgds or observation_model is None
    if do_lgds:
        print('Building linear LGDS model.')
    if not do_lgds and observation_model is None:
        print('Not perfoming LGDS but also no observation model descriptor is given! Falling back to linear model with numerical optimization.')
    if linear:
        model = torch.nn.Linear(latent_dim, observation_dim, bias=False)
        torch.nn.init.eye_(model.weight)
    else:
        print('Building observation model from descriptor %s.' % (str(observation_model) if type(observation_model) == list else f'<{observation_model}>'))
        model = util.build_dynamic_model(observation_model, latent_dim, observation_dim)
    return model


def build_result_dict(iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, observations_without_control: np.ndarray, control_inputs: Optional[np.ndarray],
                      neutral_control_input: Optional[np.ndarray], latents: np.ndarray, A: np.ndarray, B: Optional[np.ndarray], g_params: collections.OrderedDict, m0: np.ndarray,
                      y_shift, y_scale, u_shift, u_scale, Q: np.ndarray, R: np.ndarray, V0: np.ndarray, V_hat: np.ndarray, log_likelihood: Optional[float]):
    result_dict = {
        'iterations': iterations,
        'log_likelihood': log_likelihood,
        'input': {
            'observations': observations.copy(),
            'observations_noisy': observations_noisy.copy(),
            'observations_without_control': observations_without_control.copy(),
            'control_inputs': None if control_inputs is None else control_inputs.copy(),
            'neutral_control_input': None if neutral_control_input is None else neutral_control_input.copy()
        },
        'preprocessing': {
            'y_shift': None if y_shift is None else y_shift.copy(),
            'y_scale': None if y_scale is None else y_scale.copy(),
            'u_shift': None if u_shift is None else u_shift.copy(),
            'u_scale': None if u_scale is None else u_scale.copy()
        },
        'estimations': {
            'latents': latents.copy(),
            'A': A.copy(),
            'B': None if B is None else B.copy(),
            'g_params': g_params.copy(),
            'm0': m0.copy(),
            'Q': Q.copy(),
            'R': R.copy(),
            'V0': V0.copy(),
            'V_hat': V_hat.copy()
        }
    }
    return result_dict


# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, do_lgds, do_whitening, title, epsilon, max_iterations, estimate_diagonal_noise, g_optimization_learning_rate, g_optimization_precision,
         g_optimization_max_iterations, create_checkpoint_every_n_iterations, load_initialization_from_file, T_train, latent_dim):
    if title is None:
        raise ExperimentNotConfiguredInterrupt()

    observations_all, observations_all_noisy, observations_without_control, control_inputs, neutral_control_input = load_observations()
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
            # noinspection PyTypeChecker
            checkpoint = build_result_dict(iteration, observations_all, observations_all_noisy, observations_without_control, control_inputs, neutral_control_input,
                                           em.get_estimated_latents(), *em.get_estimations(), *em.get_shift_scale_data(), *em.get_covariances(), None)
            _, f_path = tempfile.mkstemp(prefix='checkpoint_%05d-' % iteration, suffix='.json')
            with open(f_path, 'w') as f:
                f.write(jsonpickle.dumps({'result': checkpoint}))
            _run.add_artifact(f_path, 'checkpoint_%05d.json' % iteration, metadata={'iteration': iteration})
            os.remove(f_path)

    initialization = EMInitialization()
    if load_initialization_from_file is not None:
        with open(load_initialization_from_file) as f:
            estimations = jsonpickle.loads(f.read())['result']['estimations']
        initialization.A = estimations['A']
        initialization.Q = estimations['Q']
        initialization.g = estimations['g_params']
        initialization.R = estimations['R']
        initialization.m0 = estimations['m0']
        initialization.V0 = estimations['V0']

    g = load_observation_model()

    options = EMOptions()
    options.do_lgds = do_lgds
    options.do_whitening = do_whitening
    options.precision = epsilon
    options.max_iterations = max_iterations
    options.estimate_diagonal_noise = estimate_diagonal_noise
    options.log = _log.info
    options.g_optimization_learning_rate = g_optimization_learning_rate
    options.g_optimization_precision = g_optimization_precision
    options.g_optimization_max_iterations = g_optimization_max_iterations
    em = EM(latent_dim, observations_train_noisy, control_inputs_train, model=g, initialization=initialization, options=options)
    log_likelihoods = em.fit(callback=callback)
    A_est, B_est, g_params_est, m0_est = em.get_estimations()
    Q_est, R_est, V0_est, V_hat_est = em.get_covariances()
    latents = em.get_estimated_latents()

    Q_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    R_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    V0_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    # noinspection PyTypeChecker
    return build_result_dict(len(log_likelihoods), observations_all, observations_all_noisy, observations_without_control, control_inputs, neutral_control_input, latents,
                             *em.get_estimations(), *em.get_shift_scale_data(), Q_est, R_est, V0_est, V_hat_est, log_likelihoods[-1])
