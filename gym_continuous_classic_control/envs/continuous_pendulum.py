from gym import spaces
from gym.envs import classic_control
import numpy as np


class NonNormalizedContinuousPendulumEnv(classic_control.PendulumEnv):
    def __init__(self, g=10.0):
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.b = 1.
        self.viewer = None

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf], dtype=np.float32),
            high=-np.array([np.pi, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.seed()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt

        u = u[0]
        self.last_u = u  # for rendering

        newth = th + thdot * dt
        newth = angle_normalize(newth)
        newthdot = thdot + (-g / l * np.sin(th + np.pi) - (b * thdot + u) / (m * l ** 2)) * dt

        self.state = np.array([newth, newthdot])
        return self.state, 0, False, {}

class ContinuousPendulumEnv(NonNormalizedContinuousPendulumEnv):
    def __init__(self, g=10.0):
        super().__init__(g)

    def step(self, u):
        obs, reward, done, info = super().step(u)
        info['state'] = self.state

        return self._get_obs(), reward, done, info

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
