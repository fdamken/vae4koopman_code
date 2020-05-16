import numpy as np
from sacred.utils import SacredInterrupt

from src.experiment import ex


EXPERIMENTS = {
        'state10d_observation6d':               (True, {
                'title': '10D State, 6D Observation, 1 Sequence',
                'T':     50,
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
                'R':     1e-5 * np.eye(6),
                'm0':    np.ones(10),
                'V0':    1e-5 * np.eye(10)
        }),
        'state10d_observation6d_multisequence': (True, {
                'title': '10D State, 6D Observation, 10 Sequences',
                'T':     50,
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
                'R':     1e-5 * np.eye(6),
                'm0':    np.ones(10),
                'V0':    1e-5 * np.eye(10),
                'N':     10
        }),

        'state1d_observation1d':                (True, {
                'title': '1D State, 1D Observation, 1 Sequence',
                'T':     50,
                'A':     np.array([[1]]),
                'Q':     np.array([[2]]),
                'C':     np.array([[1]]),
                'R':     np.array([[5]]),
                'm0':    np.array([1]),
                'V0':    np.array([[1]])
        }),
        'state1d_observation1d_multisequence':  (True, {
                'title': '1D State, 1D Observation, 10 Sequences',
                'T':     50,
                'A':     np.array([[1]]),
                'Q':     np.array([[2]]),
                'C':     np.array([[1]]),
                'R':     np.array([[5]]),
                'm0':    np.array([1]),
                'V0':    np.array([[1]]),
                'N':     10
        }),

        'state1d_observation2d':                (True, {
                'title': '1D State, 2D Observation, 1 Sequence',
                'T':     50,
                'A':     1.1 * np.eye(1),
                'Q':     1e-5 * np.eye(1),
                'C':     np.array([[1],
                                   [2]]),
                'R':     1e-5 * np.eye(2),
                'm0':    np.ones(1),
                'V0':    1e-5 * np.eye(1)
        }),
        'state1d_observation2d_multisequence':  (True, {
                'title': '1D State, 2D Observation, 10 Sequences',
                'T':     50,
                'A':     1.1 * np.eye(1),
                'Q':     1e-5 * np.eye(1),
                'C':     np.array([[1],
                                   [2]]),
                'R':     1e-5 * np.eye(2),
                'm0':    np.ones(1),
                'V0':    1e-5 * np.eye(1),
                'N':     10
        }),

        'state2d_observation2d':                (True, {
                'title': '2D State, 2D Observation, 1 Sequence',
                'T':     50,
                'A':     1.1 * np.eye(2),
                'Q':     1e-5 * np.eye(2),
                'C':     np.array([[1, 0],
                                   [0, 2]]),
                'R':     1e-5 * np.eye(2),
                'm0':    np.ones(2),
                'V0':    1e-5 * np.eye(2)
        }),
        'state2d_observation2d_multisequence':  (True, {
                'title': '2D State, 2D Observation, 10 Sequences',
                'T':     50,
                'A':     1.1 * np.eye(2),
                'Q':     1e-5 * np.eye(2),
                'C':     np.array([[1, 0],
                                   [0, 2]]),
                'R':     1e-5 * np.eye(2),
                'm0':    np.ones(2),
                'V0':    1e-5 * np.eye(2),
                'N':     10
        }),

        'state3d_observation3d':                (True, {
                'title': '3D State, 3D Observation, 1 Sequence',
                'T':     50,
                'A':     1.1 * np.eye(3),
                'Q':     1e-5 * np.eye(3),
                'C':     np.array([[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3]]),
                'R':     1e-5 * np.eye(3),
                'm0':    np.ones(3),
                'V0':    1e-5 * np.eye(3)
        }),
        'state3d_observation3d_multisequence':  (True, {
                'title': '3D State, 3D Observation, 10 Sequences',
                'T':     50,
                'A':     1.1 * np.eye(3),
                'Q':     1e-5 * np.eye(3),
                'C':     np.array([[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3]]),
                'R':     1e-5 * np.eye(3),
                'm0':    np.ones(3),
                'V0':    1e-5 * np.eye(3),
                'N':     10
        }),

        'state3d_observation6d':                (True, {
                'title': '3D State, 6D Observation, 1 Sequence',
                'T':     50,
                'A':     1.1 * np.eye(3),
                'Q':     1e-5 * np.eye(3),
                'C':     np.array([[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3],
                                   [1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3]]),
                'R':     1e-5 * np.eye(6),
                'm0':    np.ones(3),
                'V0':    1e-5 * np.eye(3)
        }),
        'state3d_observation6d_multisequence':  (True, {
                'title': '3D State, 6D Observation, 10 Sequences',
                'T':     50,
                'A':     1.1 * np.eye(3),
                'Q':     1e-5 * np.eye(3),
                'C':     np.array([[1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3],
                                   [1, 0, 0],
                                   [0, 2, 0],
                                   [0, 0, 3]]),
                'R':     1e-5 * np.eye(6),
                'm0':    np.ones(3),
                'V0':    1e-5 * np.eye(3),
                'N':     10
        })
}

if __name__ == '__main__':
    for name, (enabled, config) in EXPERIMENTS.items():
        if enabled:
            try:
                ex.run(config_updates = config, options = { '--name': name, '--debug': True })
            except SacredInterrupt:
                pass
