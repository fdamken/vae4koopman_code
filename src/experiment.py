import collections
import os
import sys
import tempfile
from typing import List, Optional, Union, Tuple

import jsonpickle
import numpy as np
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

use_neptune = os.environ.get('NO_NEPTUNE') is None
data_dir = os.environ.get('DATA_DIR', 'tmp_data')

ex = Experiment('vae-koopman')
ex.observers.append(FileStorageObserver('tmp_results'))
if use_neptune:
    ex.observers.append(NeptuneObserver(project_name='fdamken/variational-koopman'))

if not os.path.isdir(data_dir):
    raise Exception(f'Data directory {data_dir} does not exist or is not a directory!')


# noinspection PyUnusedLocal,PyPep8Naming
@ex.config
def defaults():
    seed = 42

    # General experiment description.
    title = None
    data_dir = 'tmp_data'
    data_file_pattern = '{name}.json'
    create_checkpoint_every_n_iterations = 5
    load_initialization_from_file = None
    do_whitening = False

    # Convergence checking configuration.
    epsilon = 0.00001
    max_iterations = 100
    g_optimization_learning_rate = 0.01
    g_optimization_precision = 1e-3
    g_optimization_max_iterations = 100

    # Dimensionality configuration.
    latent_dim = None

    # Observation model configuration.
    observation_model = None


# noinspection PyUnusedLocal,PyPep8Naming,DuplicatedCode
@ex.named_config
def pendulum():
    title = 'Pendulum'
    latent_dim = 10
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming,DuplicatedCode
@ex.named_config
def pendulum_damped():
    title = 'Damped Pendulum'
    do_whitening = True
    max_iterations = 200
    latent_dim = 4
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def pendulum_gym():
    title = 'Pendulum (Gym), Control'
    max_iterations = 200
    latent_dim = 4
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lunar_lander_gym():
    title = 'Lunar Lander (Gym), Control'
    max_iterations = 500
    latent_dim = 10
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def cartpole_gym():
    title = 'Cartpole (Gym), Control'
    max_iterations = 350
    latent_dim = 32
    observation_model = ['Linear(in_features, 64)', 'Tanh()', 'Linear(64, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def acrobot_gym():
    title = 'Acrobot (Gym), Control'
    max_iterations = 350
    latent_dim = 16
    observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def polynomial():
    title = 'Polynomial Koopman'
    g_optimization_max_iterations = 10000
    latent_dim = 3
    observation_model = ['Linear(in_features, 10)', 'Tanh()', 'Linear(10, out_features)']


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds():
    title = 'Simple Linear System'
    max_iterations = 100
    latent_dim = 5


# noinspection PyUnusedLocal,PyPep8Naming
@ex.named_config
def lgds_control():
    title = 'Linear System with Control'
    max_iterations = 150
    latent_dim = 2


@ex.capture(prefix='data')
def load_data(observations_train: np.ndarray, control_inputs_train: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return observations_train, control_inputs_train


@ex.capture
def load_observation_model(latent_dim: int, observation_dim_names: List[str], observation_model: Union[str, List[str]]):
    observation_dim = len(observation_dim_names)
    linear = observation_model is None
    if observation_model is None:
        print('No observation model descriptor is given! Falling back to linear model with numerical optimization.')
    if linear:
        model = torch.nn.Linear(latent_dim, observation_dim, bias=False)
        torch.nn.init.eye_(model.weight)
    else:
        print('Building observation model from descriptor %s.' % (str(observation_model) if type(observation_model) == list else f'<{observation_model}>'))
        model = util.build_dynamic_model(observation_model, latent_dim, observation_dim)
    return model


def build_result_dict(iterations: int, latents: np.ndarray, A: np.ndarray, B: Optional[np.ndarray], g_params: collections.OrderedDict, m0: np.ndarray, y_shift, y_scale, u_shift,
                      u_scale, Q: np.ndarray, R: np.ndarray, V0: np.ndarray, V_hat: np.ndarray, log_likelihood: Optional[float]):
    result_dict = {
        'iterations': iterations,
        'log_likelihood': log_likelihood,
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
@ex.main
def main(_run: Run, _log, do_whitening, title, epsilon, max_iterations, g_optimization_learning_rate, g_optimization_precision, g_optimization_max_iterations,
         create_checkpoint_every_n_iterations, load_initialization_from_file, latent_dim):
    if title is None:
        raise ExperimentNotConfiguredInterrupt()

    observations_train, control_inputs_train = load_data()

    def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)
        _run.log_scalar('g_ll', g_ll, iteration)
        _run.log_scalar('g_iterations', g_iterations, iteration)
        for i, ll in enumerate(g_ll_history):
            _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

        if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
            # noinspection PyTypeChecker
            checkpoint = build_result_dict(iteration, em.get_estimated_latents(), *em.get_estimations(), *em.get_shift_scale_data(), *em.get_covariances(), None)
            _, f_path = tempfile.mkstemp(prefix='checkpoint_%05d-' % iteration, suffix='.json')
            with open(f_path, 'w') as file:
                file.write(jsonpickle.dumps({'result': checkpoint}))
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
    options.do_whitening = do_whitening
    options.precision = epsilon
    options.max_iterations = max_iterations
    options.log = _log.info
    options.g_optimization_learning_rate = g_optimization_learning_rate
    options.g_optimization_precision = g_optimization_precision
    options.g_optimization_max_iterations = g_optimization_max_iterations
    em = EM(latent_dim, observations_train, control_inputs_train, model=g, initialization=initialization, options=options)
    log_likelihoods = em.fit(callback=callback)
    Q_est, R_est, V0_est, V_hat_est = em.get_covariances()
    latents = em.get_estimated_latents()

    Q_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    R_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    V0_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    # noinspection PyTypeChecker
    return build_result_dict(len(log_likelihoods), latents, *em.get_estimations(), *em.get_shift_scale_data(), Q_est, R_est, V0_est, V_hat_est, log_likelihoods[-1])


if __name__ == '__main__':
    args = sys.argv[1:]
    data_file_name = args[0]
    data_file_path = f'{data_dir}/{data_file_name}.json'
    if not os.path.isfile(data_file_path):
        raise Exception(f'Data file {data_file_path} does not exist or is not a file!')
    ex.add_config(data_file_path)
    sacred_args = args[1:]
    append_config_name = True
    for arg in sacred_args:
        if '=' not in arg and arg != 'with':
            append_config_name = False
            break
    if append_config_name:
        if len(sacred_args) == 0 or sacred_args[0] != 'with':
            sacred_args = ['with'] + sacred_args
        sacred_args.append(data_file_name)
    # The first argument it cut away as it's usually the name of the script.
    sacred_args = [''] + sacred_args
    ex.run_commandline(sacred_args)
