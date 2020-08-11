import collections
import os
import tempfile
from typing import Optional, Tuple

import jsonpickle
import numpy as np
import scipy as sp
import scipy.integrate as sci
import torch
from neptunecontrib.monitoring.sacred import NeptuneObserver
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run

from src.em import EM
from src.util import MatrixProblemInterrupt


ex = Experiment('pendulum')
ex.observers.append(FileStorageObserver('tmp_results'))
if os.environ.get('NO_NEPTUNE') is None:
    ex.observers.append(NeptuneObserver(project_name = 'fdamken/variational-koopman'))



# noinspection PyUnusedLocal
@ex.config
def config():
    title = 'Pendulum'
    seed = 42
    epsilon = 0.00001
    max_iterations = 100
    g_optimization_precision = 1e-3
    g_optimization_max_iterations = None
    create_checkpoint_every_n_iterations = 5
    load_initialization_from_file = None
    h = 0.1
    t_final = 50.0
    T = int(t_final / h)
    N = 1
    latent_dim = 3
    initial_value_mean = np.array([0.0872665, 0.0])
    initial_value_cov = np.diag([np.pi / 8.0, 0.0])
    R = 1e-5 * np.eye(2)



@ex.capture
def sample_dynamics(t_final: float, T: int, N: int, h: float, initial_value_mean: np.ndarray, initial_value_cov: np.ndarray, R: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    if not np.isclose(T * h, t_final):
        raise Exception('T, h and t_final are inconsistent! The result of T * h should equal t_final.')

    ode = lambda x1, x2: np.asarray([x2,
                                     np.sin(x1)])

    sequences = []
    for _ in range(0, N):
        initial_value = np.random.multivariate_normal(initial_value_mean, initial_value_cov)
        sequences.append(sci.solve_ivp(lambda t, x: ode(*x), (0, t_final), initial_value, t_eval = np.arange(0, t_final, h), method = 'Radau').y.T)
    sequences = np.asarray(sequences)
    sequences_noisy = sequences + np.random.multivariate_normal(np.zeros(np.prod(sequences.shape)), sp.linalg.block_diag(*([R] * N * T))).reshape(sequences.shape)
    return sequences, sequences_noisy



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



def build_result_dict(iterations: int, observations: np.ndarray, observations_noisy: np.ndarray, latents: np.ndarray, A: np.ndarray, Q: np.ndarray,
                      g_params: collections.OrderedDict, R: np.ndarray, m0: np.ndarray, V0: np.ndarray, final_log_likelihood: Optional[float]):
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
    if final_log_likelihood is not None:
        result_dict['log_likelihood'] = final_log_likelihood
    return result_dict



# noinspection PyPep8Naming
@ex.automain
def main(_run: Run, _log, epsilon: Optional[float], max_iterations: Optional[int], create_checkpoint_every_n_iterations: int, load_initialization_from_file: Optional[str],
         g_optimization_precision: float, g_optimization_max_iterations: Optional[int], latent_dim: int):
    observations, observations_noisy = sample_dynamics()
    state_dim = observations.shape[2]


    def callback(iteration, log_likelihood, g_ll, g_iterations, g_ll_history):
        if log_likelihood is not None:
            _run.log_scalar('log_likelihood', log_likelihood, iteration)
        _run.log_scalar('g_ll', g_ll, iteration)
        _run.log_scalar('g_iterations', g_iterations, iteration)
        for i, ll in enumerate(g_ll_history):
            _run.log_scalar('g_ll_history_%05d' % iteration, ll, i)

        if iteration == 1 or iteration % create_checkpoint_every_n_iterations == 0:
            A_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp = em.get_estimations()
            checkpoint = build_result_dict(iteration, observations, observations_noisy, em.get_estimated_states(), A_cp, Q_cp, g_params_cp, R_cp, m0_cp, V0_cp, None)
            _, f_path = tempfile.mkstemp(prefix = 'checkpoint_%05d-' % iteration, suffix = '.json')
            with open(f_path, 'w') as f:
                f.write(jsonpickle.dumps({ 'result': checkpoint }))
            _run.add_artifact(f_path, 'checkpoint_%05d.json' % iteration, metadata = { 'iteration': iteration })
            os.remove(f_path)


    A_init, Q_init, g_init, R_init, m0_init, V0_init = [None] * 6
    if load_initialization_from_file is not None:
        with open(load_initialization_from_file) as f:
            initialization = jsonpickle.loads(f.read())['result']['estimations']
        A_init = initialization['A']
        Q_init = initialization['Q']
        g_init = initialization['g_params']
        R_init = initialization['R']
        m0_init = initialization['m0']
        V0_init = initialization['V0']

    g = Model(latent_dim, state_dim)
    em = EM(latent_dim, observations, model = g, A_init = A_init, Q_init = Q_init, g_init = g_init, R_init = R_init, m0_init = m0_init, V0_init = V0_init)
    log_likelihoods = em.fit(epsilon, max_iterations = max_iterations,
                             log = _log.info,
                             callback = callback,
                             g_optimization_precision = g_optimization_precision,
                             g_optimization_max_iterations = g_optimization_max_iterations)
    A_est, Q_est, g_params_est, R_est, m0_est, V0_est = em.get_estimations()
    latents = em.get_estimated_states()

    # Collect results and metrics.
    Q_problem, R_problem, V0_problem = em.get_problems()
    final_log_likelihood = log_likelihoods[-1]
    iterations = len(log_likelihoods)

    if Q_problem or R_problem or V0_problem:
        raise MatrixProblemInterrupt()

    # Return the results.
    return build_result_dict(iterations, observations, observations_noisy, latents, A_est, Q_est, g_params_est, R_est, m0_est, V0_est, final_log_likelihood)
