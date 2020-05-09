import numpy as np

from src.experiment import ex


EXPERIMENTS = {
        'state3d_observation6d': (True, {
                'title': '3D State, 6D Observation',
                'T':     50,
                'pi1':   np.ones(3),
                'V1':    1e-5 * np.eye(3),
                'A':     1.1 * np.eye(3),
                'Q':     1e-5 * np.eye(3),
                'C':     np.array([[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3],
                                   [1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3]]),
                'R':     1e-5 * np.eye(6)
        }),
        'state10d_observation6d': (False, {
                'title': '10D State, 6D Observation',
                'T':     50,
                'pi1':   np.ones(10),
                'V1':    1e-5 * np.eye(10),
                'A':     1.1 * np.eye(10),
                'Q':     1e-5 * np.eye(10),
                'C':     np.array([[1, 0, 0, 0, 0, 0],
                                   [0, 2, 0, 0, 0, 0],
                                   [0, 0, 3, 0, 0, 0],
                                   [0, 0, 0, 4, 0, 0],
                                   [1, 0, 0, 0, 5, 0],
                                   [0, 2, 0, 0, 0, 6],
                                   [0, 0, 3, 0, 0, 0],
                                   [0, 0, 0, 4, 0, 0],
                                   [0, 0, 0, 0, 5, 0],
                                   [0, 0, 0, 0, 0, 6]]).T,
                'R':     1e-5 * np.eye(6)
        }),
        'state2d_observation2d':  (False, {
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
        # This does not work without initialization via factor analysis.
        'state2d_observation1d':  (False, {
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
        'state1d_observation2d':  (False, {
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
        'state1d_observation1d':  (False, {
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
