import collections
import json
import warnings
from typing import List, Optional, Tuple, Union

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch

from src import util


jsonpickle_numpy.register_handlers()



class ExperimentConfig:
    def __init__(self, do_lgds: bool, title: str, h: float, t_final: float, T: int, T_train: int, N: int, latent_dim: int, observation_dim: int, observation_dim_names: List[str],
                 observation_model: Union[str, List[str]]):
        # General experiment description.
        self.title = title
        # Do regular LGDS instead of nonlinear measurements?
        self.do_lgds = do_lgds
        # Sequence configuration (time span and no. of sequences).
        self.h = h
        self.t_final = t_final
        self.t_final_train = T_train * h
        self.T = T
        self.T_train = T_train
        self.N = N
        # Dimensionality configuration.
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.observation_dim_names = observation_dim_names
        #  Observation model configuration.
        self.observation_model = observation_model



class ExperimentResult:
    def __init__(self, config: ExperimentConfig, iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, observations_without_control: np.ndarray,
                 control_inputs: Optional[np.ndarray], neutral_control_input: Optional[np.ndarray], estimations_latents: np.ndarray, A: np.ndarray, B: Optional[np.ndarray],
                 g_params: collections.OrderedDict, m0: np.ndarray, Q: np.ndarray, R: np.ndarray, V0: np.ndarray, V_hat: np.ndarray):
        self.iterations = iterations
        self.observations = observations
        self.observations_noisy = observations_noisy
        self.observations_without_control = observations_without_control
        self.observations_train = self.observations[:config.T_train]
        self.observations_test = self.observations[config.T_train:]
        self.control_inputs = control_inputs
        self.control_inputs_train = None if self.control_inputs is None else self.control_inputs[:config.T_train]
        self.control_inputs_test = None if self.control_inputs is None else self.control_inputs[config.T_train:]
        self.neutral_control_input = neutral_control_input
        self.estimations_latents = estimations_latents
        self.A = A
        self.B = B
        if config.do_lgds:
            self.g = torch.nn.Linear(config.latent_dim, config.observation_dim, bias = False)
        else:
            self.g = util.build_dynamic_model(config.observation_model, config.latent_dim, config.observation_dim)
        self.g.load_state_dict(g_params)
        self.m0 = m0
        self.Q = Q
        self.R = R
        self.V0 = V0
        self.V_hat = V_hat


    def g_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.g(torch.tensor(x, dtype = torch.float32)).detach().cpu().numpy()



class ExperimentMetrics:
    def __init__(self, log_likelihood: List[float], g_iterations: List[int], g_final_log_likelihood: List[float]):
        self.log_likelihood = log_likelihood
        self.g_iterations = g_iterations
        self.g_final_log_likelihood = g_final_log_likelihood



def load_run(result_dir: str, result_file: str, metrics_file: Optional[str] = None) \
        -> Union[Tuple[ExperimentConfig, ExperimentResult], Tuple[ExperimentConfig, ExperimentResult, ExperimentMetrics]]:
    with open('%s/config.json' % result_dir) as f:
        config_dict = jsonpickle.loads(f.read())
        do_lgds = config_dict['do_lgds'] if 'do_lgds' in config_dict else False
        config = ExperimentConfig(do_lgds, config_dict['title'], config_dict['h'], config_dict['t_final'], config_dict['T'], config_dict['T_train'], config_dict['N'],
                                  config_dict['latent_dim'], config_dict['observation_dim'], config_dict['observation_dim_names'], config_dict['observation_model'])

    with open('%s/%s.json' % (result_dir, result_file)) as f:
        result_dict = jsonpickle.loads(f.read())['result']
        input_dict = result_dict['input']
        estimations_dict = result_dict['estimations']
        control_inputs = input_dict['control_inputs'] if 'control_inputs' in input_dict else None
        B = estimations_dict['B'] if 'B' in estimations_dict else None
        neutral_control_input = input_dict['neutral_control_input'] if 'neutral_control_input' in input_dict else None
        if (control_inputs is None) != (B is None):
            raise Exception('Inconsistent experiment result! Both control_inputs and B must either be an numpy.ndarray or None.')
        result = ExperimentResult(config, result_dict['iterations'], input_dict['observations'], input_dict['observations_noisy'], input_dict['observations_without_control'],
                                  control_inputs, neutral_control_input, estimations_dict['latents'], estimations_dict['A'], B, estimations_dict['g_params'],
                                  estimations_dict['m0'], estimations_dict['Q'], estimations_dict['R'], estimations_dict['V0'], estimations_dict['V_hat'])

    if metrics_file is None:
        metrics = None
    else:
        with open('%s/%s.json' % (result_dir, metrics_file)) as f:
            metrics_dict = json.load(f)
            log_likelihood = metrics_dict['log_likelihood']['values']
            g_iterations = metrics_dict['g_iterations']['values']
            g_final_log_likelihood = metrics_dict['g_ll']['values']
            metrics = ExperimentMetrics(log_likelihood, g_iterations, g_final_log_likelihood)

    return config, result, metrics



def normalize_covariances(covariances: np.ndarray) -> np.ndarray:
    """
    "Removes" values close to zero to avoid invalid square-roots with negative
    numbers due to numerical instabilities. A value slightly more than zero
    (e.g. 1e-8) does not remove the covariance at all resulting in a weird
    plot, but removes just enough to visualize that the model is extremely
    confident while not looking weird.

    :param covariances: The covariances to modify.
    :return: The modified matrix. However, the references array is also
             changed. Just for convenience.
    """

    threshold = -0.01
    if (covariances < threshold).any():
        warnings.warn('Found covariances that are too negative!\n\tThreshold: %f\n\tCovariances in question: %s' % (threshold, str(list(covariances[covariances < threshold]))))
    covariances[covariances < 0] = 1e-8
    return covariances
