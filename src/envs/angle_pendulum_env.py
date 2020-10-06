import numpy as np
from gym.envs.classic_control import PendulumEnv



class PendulumAngleEnv(PendulumEnv):
    def __init__(self):
        super().__init__()


    def _get_obs(self):
        theta, theta_dot = self.state
        return np.asarray([theta, theta_dot])
