from gym.envs.classic_control import CartPoleEnv


class UncontrolledCartPole(CartPoleEnv):
    def __init__(self):
        super().__init__()

        self.force_mag = 0.0
