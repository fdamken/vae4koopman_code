import collections
import os
import sys
import tempfile
from typing import List, Optional, Union, Tuple, Dict

import jsonpickle
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src import util
from src.em import EM, EMInitialization, EMOptions
from src.util import ExperimentNotConfiguredInterrupt, MatrixProblemInterrupt

torch.set_default_dtype(torch.double)
util.apply_sacred_frame_error_workaround()

ENV_RESULTS_DIR = os.environ.get('RESULTS_DIR')


def run_experiment(data_file_name: str, sacred_args: Optional[List[str]] = None, config_updates: Optional[Dict[str, Union[str, int, float]]] = None,
                   results_dir: str = 'tmp_results', dry_run: bool = False, debug: bool = False):
    if sacred_args is None:
        sacred_args = []
    if config_updates is None:
        config_updates = {}
    if ENV_RESULTS_DIR is not None:
        results_dir = ENV_RESULTS_DIR

    ex = Experiment('vae-koopman')

    print(f'PRE-SACRED: Configuring file storage observer for results dir <{results_dir}>.')
    ex.observers.append(FileStorageObserver(results_dir))

    data_dir = os.environ.get('DATA_DIR', 'tmp_data')
    if not os.path.isdir(data_dir):
        raise Exception(f'Data directory <{data_dir}> does not exist or is not a directory!')
    data_file_path = f'{data_dir}/{data_file_name}.json'
    if not os.path.isfile(data_file_path):
        raise Exception(f'Data file {data_file_path} does not exist or is not a file!')

    append_config_name = True
    sacred_args = sacred_args.copy()
    for arg in sacred_args:
        if '=' not in arg and arg != 'with':
            append_config_name = False
            break
    if append_config_name:
        if len(sacred_args) == 0 or sacred_args[0] != 'with':
            sacred_args = ['with'] + sacred_args
        sacred_args.append(data_file_name)
    for key, value in config_updates.items():
        print(f'PRE-SACRED: Updating sacred arguments with config from dictionary: {key}={value}')
        sacred_args.append(f'{key}={value}')
    print(f'PRE-SACRED: Expanded arguments to run sacred with {sacred_args}.')
    # The first argument it cut away as it's usually the name of the script.
    sacred_args = [''] + sacred_args
    if debug:
        sacred_args.append('--debug')

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.config
    def defaults():
        seed = 42
        use_cuda = False

        # General experiment description.
        title = None
        data_dir = 'tmp_data'
        data_file_pattern = '{name}.json'
        create_checkpoint_every_n_iterations = 5
        load_initialization_from_file = None
        do_whitening = False
        observation_noise_cov = 0.0  # Observation noise can be used for regularization.

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
        latent_dim = 10
        observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def pendulum_gym():
        title = 'Pendulum (Gym)'
        max_iterations = 500
        latent_dim = 10
        observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def lunar_lander_gym():
        title = 'Lunar Lander (Gym)'
        max_iterations = 500
        latent_dim = 10
        observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def cartpole_gym():
        title = 'Cartpole (Gym)'
        do_whitening = True
        max_iterations = 500
        latent_dim = 8
        observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def acrobot_gym():
        title = 'Acrobot (Gym)'
        max_iterations = 250
        latent_dim = 16
        observation_model = ['Linear(in_features, 50)', 'Tanh()', 'Linear(50, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def polynomial():
        title = 'Polynomial Koopman'
        max_iterations = 200
        latent_dim = 3
        observation_model = ['Linear(in_features, out_features)']

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def lgds():
        title = 'Simple Linear System'
        max_iterations = 100
        latent_dim = 2

    # noinspection PyUnusedLocal,PyPep8Naming
    @ex.named_config
    def lgds_control():
        title = 'Linear System with Control'
        max_iterations = 150
        latent_dim = 2

    @ex.capture(prefix='data')
    def _load_data(observations_train: np.ndarray, control_inputs_train: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return observations_train, control_inputs_train

    @ex.capture
    def _load_observation_model(_log, latent_dim: int, observation_dim_names: List[str], observation_model: Union[str, List[str]]):
        observation_dim = len(observation_dim_names)
        linear = observation_model is None
        if observation_model is None:
            _log.info('No observation model descriptor is given! Falling back to linear model with numerical optimization.')
        if linear:
            model = torch.nn.Linear(latent_dim, observation_dim, bias=False)
            torch.nn.init.eye_(model.weight)
        else:
            _log.info('Building observation model from descriptor %s.' % (str(observation_model) if type(observation_model) == list else f'<{observation_model}>'))
            model = util.build_dynamic_model(observation_model, latent_dim, observation_dim)
        return model

    def _build_result_dict(iterations: int, observations_train_noisy: np.ndarray, latents: np.ndarray, A: np.ndarray, B: Optional[np.ndarray],
                           g_params: collections.OrderedDict, m0: np.ndarray, y_shift, y_scale, u_shift, u_scale, Q: np.ndarray, R: np.ndarray, V0: np.ndarray, V_hat: np.ndarray,
                           log_likelihood: Optional[float]):
        result_dict = {
            'iterations': iterations,
            'log_likelihood': log_likelihood,
            'observations_train_noisy': observations_train_noisy.copy(),
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
    def _main(_run: Run, _log, use_cuda, do_whitening, title, epsilon, max_iterations, observation_noise_cov, g_optimization_learning_rate, g_optimization_precision,
              g_optimization_max_iterations, create_checkpoint_every_n_iterations, load_initialization_from_file, latent_dim):
        if title is None:
            raise ExperimentNotConfiguredInterrupt()

        observations_train_noisy, control_inputs_train = _load_data()
        observations_train_noisy += np.random.multivariate_normal(np.array([0.0]), np.array([[observation_noise_cov]]), size=observations_train_noisy.shape).reshape(
            observations_train_noisy.shape)

        def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
            if log_likelihood is not None:
                _run.log_scalar('log_likelihood', log_likelihood, iteration)
            _run.log_scalar('g_ll', g_ll, iteration)
            _run.log_scalar('g_iterations', g_iterations, iteration)
            for i, ll in enumerate(g_ll_history):
                _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

            if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
                # noinspection PyTypeChecker
                checkpoint = _build_result_dict(iteration, observations_train_noisy, em.get_estimated_latents(), *em.get_estimations(), *em.get_shift_scale_data(),
                                                *em.get_covariances(), None)
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

        g = _load_observation_model()
        options = EMOptions()
        options.do_whitening = do_whitening
        options.precision = epsilon
        options.max_iterations = max_iterations
        options.log = _log.info
        options.g_optimization_learning_rate = g_optimization_learning_rate
        options.g_optimization_precision = g_optimization_precision
        options.g_optimization_max_iterations = g_optimization_max_iterations
        em = EM(latent_dim, observations_train_noisy, control_inputs_train, use_cuda, model=g, initialization=initialization, options=options)
        log_likelihoods = em.fit(callback=callback)
        Q_est, R_est, V0_est, V_hat_est = em.get_covariances()
        latents = em.get_estimated_latents()

        Q_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
        R_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
        V0_problem = ((np.linalg.eigvals(Q_est)) <= 0).any()
        if Q_problem or R_problem or V0_problem:
            raise MatrixProblemInterrupt()

        # noinspection PyTypeChecker
        return _build_result_dict(len(log_likelihoods), observations_train_noisy, latents, *em.get_estimations(), *em.get_shift_scale_data(), Q_est, R_est, V0_est, V_hat_est,
                                  log_likelihoods[-1])

    print(f'PRE-SACRED: Updating configuration with data from <{data_file_path}>.')
    ex.add_config(data_file_path)
    print(f'PRE-SACRED: Experiment setup successful. Starting experiment.')
    if dry_run:
        print(f'PRE-SACRED: This is a dry run. Quitting now!')
        return True
    return ex.run_commandline(sacred_args)


if __name__ == '__main__':
    run_experiment(sys.argv[1], sys.argv[2:])
