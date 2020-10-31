import numpy as np
from gym.envs.classic_control import AcrobotEnv


class ModifiedAcrobotEnv(AcrobotEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        position = self.np_random.normal(0.0, np.pi / 2.0, size=(2,))
        velocity = self.np_random.uniform(low=-0.1, high=0.1, size=(2,))
        self.state = np.concatenate([position, velocity], axis=0)
        return self._get_ob()
