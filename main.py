import logging

import numpy as np

from src.experiment import ex


EXPERIMENTS = {
        'state2d_observation2d': (True, {
                'title': '2D State, 2D Observation',
                'T':     50,
                'pi1':   np.array([1, 1]),
                'V1':    np.array([[1e-5, 0],
                                   [0, 1e-5]]),
                'A':     np.array([[1.1, 0],
                                   [0, 1.1]]),
                'Q':     np.array([[1e-5, 0],
                                   [0, 1e-5]]),
                'C':     np.array([[1, 0],
                                   [0, 2]]),
                'R':     np.array([[1e-5, 0],
                                   [0, 1e-5]])
        }),
        'state2d_observation1d': (False, {
                'title':   '2D State, 1D Observation',
                'enabled': False,
                'T':       10,
                'pi1':     np.array([0, 0]),
                'V1':      np.array([[1, 0],
                                     [0, 1]]),
                'A':       np.array([[1, 0],
                                     [0, 1]]),
                'Q':       np.array([[1, 0],
                                     [0, 1]]),
                'C':       np.array([[1, 0]]),
                'R':       np.array([[1]])
        }),
        'state1d_observation2d': (False, {
                'title': '1D State, 2D Observation',
                'T':     100,
                'pi1':   np.array([0]),
                'V1':    np.array([[1]]),
                'A':     np.array([[1]]),
                'Q':     np.array([[2]]),
                'C':     np.array([[1],
                                   [2]]),
                'R':     np.array([[2, 0],
                                   [0, 2]])
        }),
        'state1d_observation1d': (False, {
                'title': '1D State, 1D Observation',
                'T':     100,
                'pi1':   np.array([1]),
                'V1':    np.array([[1]]),
                'A':     np.array([[1]]),
                'Q':     np.array([[2]]),
                'C':     np.array([[1]]),
                'R':     np.array([[5]])
        })
}

if __name__ == '__main__':
    for name, (enabled, config) in EXPERIMENTS.items():
        if enabled:
            ex.run(config_updates = config, options = { '--name': name })
