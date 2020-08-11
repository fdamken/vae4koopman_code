import collections
import json
from typing import List, Optional, Tuple, Union

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch

from src.util import WhitenedModel


jsonpickle_numpy.register_handlers()



class ExperimentConfig:
    def __init__(self, title: str, N: int, T: int, h: float, t_final: float, latent_dim: int, observation_dim: int):
        self.title = title
        self.N = N
        self.T = T
        self.h = h
        self.t_final = t_final
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim



class Model(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self._pipe = torch.nn.Sequential(
                torch.nn.Linear(in_features, 50),
                torch.nn.Tanh(),
                torch.nn.Linear(50, out_features)
        )


    def forward(self, x):
        return self._pipe(x)



class ExperimentResult:
    def __init__(self, config: ExperimentConfig, iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, estimations_latents: np.ndarray, A: np.ndarray,
                 Q: np.ndarray, g_params: collections.OrderedDict, R: np.ndarray, m0: np.ndarray, V0: np.ndarray):
        self.iterations = iterations
        self.observations = observations
        self.observations_noisy = observations_noisy
        self.estimations_latents = estimations_latents
        self.A = A
        self.Q = Q
        self.g = WhitenedModel(Model(config.latent_dim, config.observation_dim), config.latent_dim)
        self.g.load_state_dict(g_params)
        self.R = R
        self.m0 = m0
        self.V0 = V0


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
        config = ExperimentConfig(config_dict['title'], config_dict['N'], config_dict['T'], config_dict['h'], config_dict['t_final'], config_dict['latent_dim'], 2)
    with open('%s/%s.json' % (result_dir, result_file)) as f:
        result_dict = jsonpickle.loads(f.read())['result']
        input_dict = result_dict['input']
        estimations_dict = result_dict['estimations']
        result = ExperimentResult(config, result_dict['iterations'], input_dict['observations'], input_dict['observations_noisy'], estimations_dict['latents'],
                                  estimations_dict['A'], estimations_dict['Q'], estimations_dict['g_params'], estimations_dict['R'], estimations_dict['m0'], estimations_dict['V0'])
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
