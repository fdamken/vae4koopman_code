import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import gym
import jsonpickle
import numpy as np
import progressbar
import scipy.integrate as sci
import sympy as sp
import torch
from progressbar import Bar, ETA, Percentage
from sacred import Experiment
from sacred.observers import FileStorageObserver

from src import util

util.apply_sacred_frame_error_workaround()
torch.set_default_dtype(torch.double)

ex = Experiment('data-generation')
ex.observers.append(FileStorageObserver('tmp_results_data_generation'))


# noinspection PyUnusedLocal,PyPep8Naming
@ex.config
def defaults():
    # General experiment description.
    name = None
    seed = 42
    force = False
    out_dir = 'tmp_data'
    out_file_name_pattern = '{name}.json'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    observation_dim = None
    observation_dim_names = []
    dynamics_control_inputs_dim = 0

    # Dynamics sampling configuration.
    dynamics_mode = 'ode'  # Can be 'ode', 'image', 'manual' or 'gym'.
    dynamics_ode = None
    dynamics_params = {}
    dynamics_control_inputs = None
    dynamics_neutral_control = None
    initial_value_mean = None
    initial_value_cov = None
    dynamics_transform = None
    # Alternatively, the observations can be provided directly.
    dynamics_obs = None
    dynamics_obs_without_actions = None
    dynamics_manual_control_inputs = None
    dynamics_manual_neutral_control = None
    # Alternatively, the observations can be generated from a gym environment.
    gym_do_control = False  # Control does not work on most Gym environments, so do not perform control by default.
    gym_environment = None
    gym_neutral_action = None
    gym_render = False


# noinspection PyUnusedLocal,PyPep8Naming,DuplicatedCode
@ex.named_config
def pendulum():
    # General experiment description.
    name = 'pendulum'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = T // 2
    N = 1

    # Dimensionality configuration.
    observation_dim = 2
    observation_dim_names = [r'$\theta$', r'$\dot{\theta}$']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1)']
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])


# noinspection PyUnusedLocal,PyPep8Naming,DuplicatedCode
@ex.named_config
def pendulum_damped():
    # General experiment description.
    name = 'pendulum_damped'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 2 * 50.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    observation_dim = 2
    observation_dim_names = [r'$\theta$', r'\dot{\theta}']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', 'sin(x1) - d * x2']
    dynamics_params = {'d': 0.1}
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_gym():
    # General experiment description.
    name = 'pendulum_gym'

    # Sequence configuration (time span and no. of sequences).
    h = 0.05
    T_train = 50
    T = T_train * 2
    t_final = T * h
    N = 1

    # Dimensionality configuration.
    observation_dim = 3
    observation_dim_names = [r'$\cos(\theta)$', r'$\sin(\theta)$', r'$\dot{\theta}$']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'Pendulum-v0'
    gym_neutral_action = np.array([0.0])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lunar_lander_gym():
    # General experiment description.
    name = 'lunar_lander_gym'

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T * h
    N = 10

    # Dimensionality configuration.
    observation_dim = 8
    observation_dim_names = [r'$x$', r'$y$', r'$\dot{x}$', r'$\dot{y}$', r'$\theta$', r'$\dot{\theta}}$']

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
    name = 'cartpole_gym'

    # Sequence configuration (time span and no. of sequences).
    h = 0.02
    T_train = 150
    T = T_train * 2
    t_final = T * h
    N = 1

    # Dimensionality configuration.
    observation_dim = 4
    observation_dim_names = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'UncontrolledCartPole-v0'
    gym_neutral_action = 1  # The neutral action doesn't matter, the uncontrolled cart pole environment has 0 force anyway.


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def acrobot_gym():
    # General experiment description.
    name = 'acrobot_gym'

    # Sequence configuration (time span and no. of sequences).
    h = 0.2
    t_final = 15.0
    T = int(t_final / h)
    T_train = int(T * 0.75)
    N = 1

    # Dimensionality configuration.
    observation_dim = 6
    observation_dim_names = [r'$\cos(\varphi_1)$', r'$\sin(\varphi_1)$', r'$\cos(\varphi_2)$', r'$\sin(\varphi_2)$', r'$\dot{\varphi}_1$', r'$\dot{\varphi}_2$']

    # Dynamics sampling configuration.
    dynamics_mode = 'gym'
    # Alternatively, the observations can be generated from a gym environment.
    gym_environment = 'ModifiedAcrobot-v0'
    gym_neutral_action = 1


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def polynomial():
    # General experiment description.
    name = 'polynomial'

    # Sequence configuration (time span and no. of sequences).
    h = 0.02
    t_final = 2 * 1.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    observation_dim = 2
    observation_dim_names = [r'$x$', r'$\dot{x}$']

    # Dynamics sampling configuration.
    dynamics_ode = ['mu * x1', 'L * (x2 - x1 ** 2)']
    dynamics_params = {'mu': -0.05, 'L': -1.0}
    initial_value_mean = np.array([0.3, 0.4])
    initial_value_cov = np.diag([0.1, 0.1])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds():
    # General experiment description.
    name = 'lgds'

    # Sequence configuration (time span and no. of sequences).
    h = 0.1
    t_final = 24.0
    T = int(t_final / h)
    T_train = int(T / 2)
    N = 1

    # Dimensionality configuration.
    observation_dim = 2
    observation_dim_names = [r'$x_1$', r'$x_2$']

    # Dynamics sampling configuration.
    dynamics_ode = ['x2', '-x1']
    dynamics_params = {}
    initial_value_mean = np.array([0.1, 0.2])
    initial_value_cov = np.diag([1e-5, 1e-5])


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds_control():
    # General experiment description.
    name = 'lgds_control'

    # Sequence configuration (time span and no. of sequences).
    h = 1.0
    T = 200
    T_train = 150
    t_final = T / h
    N = 5

    # Dimensionality configuration.
    observation_dim = 2
    observation_dim_names = [r'$x_1$', r'$x_2$']
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

    def control_law(n_p: int, i_p: int, t_p: float, x_p: np.ndarray) -> np.ndarray:
        if dynamics_control_inputs is None:
            return np.array([])
        if callable(dynamics_control_inputs):
            return dynamics_control_inputs(n_p, t_p, x_p)
        if type(dynamics_control_inputs) == list or type(dynamics_control_inputs) == np.ndarray:
            return dynamics_control_inputs[n_p][i_p]
        raise Exception('Data type of control inputs not understood: %s' % str(type(dynamics_control_inputs)))

    sp_observation_params = ' '.join(['x%d' % i for i in range(1, observation_dim + 1)])
    sp_control_inputs_params = ' '.join(['u%d' % i for i in range(1, dynamics_control_inputs_dim + 1)])
    sp_params = sp.symbols('t %s %s' % (sp_observation_params, sp_control_inputs_params))
    ode_expr = [sp.lambdify(sp_params, sp.sympify(ode).subs(dynamics_params), 'numpy') for ode in dynamics_ode]
    ode = lambda t_p, x_p, u_p: np.asarray([expr(t_p, *x_p, *u_p) for expr in ode_expr])
    sequences = []
    sequences_without_actions = []
    sequences_actions = []
    # noinspection PyUnresolvedReferences
    for n in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        if dynamics_control_inputs is None:
            # If we don't have control inputs, we can use more sophisticated integration methods.
            solution = sci.solve_ivp(lambda t_p, x_p: ode(t_p, x_p, []), (0, t_final), initial_value, t_eval=np.arange(0, t_final, h), method='Radau')
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
def sample_gym(T: int, T_train: int, N: int, gym_do_control: bool, gym_environment: str, gym_neutral_action: np.ndarray, gym_render: bool, seed: int) \
        -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert gym_do_control is not None, 'gym_do_control is not given!'
    assert gym_environment is not None, 'gym_environment is not given!'
    assert T == T_train or gym_neutral_action is not None, 'gym_neutral_action is not given, but test data exists!'

    # Create controlled and uncontrolled environments.
    env = gym.make(gym_environment)
    env_without_control = gym.make(gym_environment)
    if gym_environment.startswith('CartPole-') and gym_do_control is False:
        env.force_mag = 0.0
        env_without_control.force_mag = 0.0
    env.seed(seed)
    env_without_control.seed(seed)
    env.action_space.seed(seed)
    env_without_control.action_space.seed(seed)
    sequences = []
    sequences_without_control = []
    sequences_actions = []
    bar = progressbar.ProgressBar(widgets=['Sampling Gym: ', Percentage(), ' ', Bar(), ' ', ETA()], maxval=N * T).start()
    for n in range(N):
        sequence = []
        sequence_without_control = []
        sequence_actions = []

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
def load_observations(dynamics_mode: str, h: float, t_final: float, T: int, T_train: int, observation_dim: int, dynamics_control_inputs_dim: int, dynamics_params: Dict[str, float],
                      dynamics_transform: List[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    assert np.isclose(T * h, t_final), 'h, t_final and T are inconsistent! Result of T * h must equal t_final.'
    assert T_train <= T, 'T_train must be less or equal to T!'
    assert dynamics_mode is not None, 'dynamics_mode is not given!'
    assert dynamics_mode in ('ode', 'image', 'manual', 'gym'), 'dynamics_mode is not one of "ode", "image" or "gym"!'

    if dynamics_mode == 'ode':
        sequences, sequences_without_actions, sequences_actions, neutral_action = sample_ode()
    elif dynamics_mode == 'gym':
        sequences, sequences_without_actions, sequences_actions, neutral_action = sample_gym()
    else:
        raise Exception(f'Invalid dynamics mode {dynamics_mode}!')
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
    return sequences, sequences_without_actions, sequences_actions, neutral_action


@ex.capture
def save_data(observations_all, observations_without_control, control_inputs, neutral_control_input, observations_train, control_inputs_train, _run, _log, name: str, out_dir: str,
              out_file_name_pattern: str, force: bool) -> str:
    file_name = out_file_name_pattern.replace('{name}', name)
    if not os.path.isdir(out_dir):
        if os.path.exists(out_dir):
            raise Exception(f'Output directory {out_dir} exists but is not a directory!')
        os.makedirs(out_dir)
    file_path = f'{out_dir}/{file_name}'
    if os.path.exists(file_path):
        if not os.path.isfile(file_path):
            raise Exception(f'Output file {file_path} exists but is no file!')
        if force:
            _log.info(f'Output file {file_path} exists but generation is enforced. Deleting.')
            os.remove(file_path)
        else:
            raise Exception(f'Warning: Output file {file_path} exists, but generation is not enforced.')
    with open(file_path, 'w') as file:
        file.write(jsonpickle.dumps({
            **_run.config,
            'data': {
                'observations': observations_all.copy(),
                'observations_without_control': observations_without_control.copy(),
                'control_inputs': None if control_inputs is None else control_inputs.copy(),
                'neutral_control_input': None if neutral_control_input is None else neutral_control_input.copy(),

                'observations_train': observations_train,
                'control_inputs_train': control_inputs_train
            }
        }))
    return file_path


@ex.main
def main(_log, T_train: int):
    observations_all, observations_without_control, control_inputs, neutral_control_input = load_observations()
    observations_train = observations_all[:, :T_train, :]
    control_inputs_train = None if control_inputs is None else control_inputs[:, :T_train - 1, :]  # The last state does not have an action.
    file_path = save_data(observations_all, observations_without_control, control_inputs, neutral_control_input, observations_train, control_inputs_train)
    _log.info(f'Stored generated data at {file_path}.')


if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv:
        ex.run_commandline()
    else:
        print('No named config specified. Running all named configurations enforcing data generation.')
        for named_config in ex.named_configs.keys():
            ex.run(named_configs=[named_config], config_updates={'force': True})
