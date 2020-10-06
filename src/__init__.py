import gym

from src.envs.angle_pendulum_env import PendulumAngleEnv


gym.envs.register(id = 'PendulumAngle-v0', entry_point = 'src.envs.angle_pendulum_env:PendulumAngleEnv')
