import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class NonNormalizedContinuousCarEnv(gym.Env):
    """
    Description:
        Simple 2D Car dynamics
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       X Position                -Inf                    Inf
        1       Y Position                -Inf                    Inf
        2       Heading Angle             -Pi rad (-180 deg)      Pi rad (180 deg)
        3       Forward Velocity          -Inf                    Inf
    Actions:
        Type: Box(1)
        Num   Action                      Min                     Max
        0     Acceleraton                 -Inf                    Inf
        1     Steering Rate               -Inf                    Inf
    Reward:
        None
    """

    def __init__(self, dt=0.02):
        self.dt = dt

        action_high = np.array([np.finfo(np.float32).max,
                                np.finfo(np.float32).max], dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.pi,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x, y, theta, v = self.state
        u_theta, u_v = action
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        x_dot = v * costheta
        y_dot = v * sintheta
        theta_dot = v * u_theta
        v_dot = u_v

        x = x + self.dt * x_dot
        y = y + self.dt  * y_dot
        theta = theta + self.dt * theta_dot
        v = v + self.dt * v_dot

        theta = angle_normalize(theta)
        self.state = (x, y, theta, v)

        return np.array(self.state), 0, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

class ContinuousCarEnv(NonNormalizedContinuousCarEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['state'] = np.array(self.state)

        return self._get_obs(), reward, done, info

    def reset(self):
        self.state = super().reset()
        return self._get_obs()

    def _get_obs(self):
        x, y, theta, v = self.state
        return np.array([x, y, np.cos(theta), np.sin(theta), v])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
