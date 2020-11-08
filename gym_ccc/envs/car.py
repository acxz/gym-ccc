"""Continuous Car Env."""
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class NonNormContCarEnv(gym.Env):
    """
    Continuous car that outpus the state as the observation.

    Description:
        Simple 2D Car dynamics
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       X Position                -Inf                    Inf
        1       Y Position                -Inf                    Inf
        2       Heading Angle             -Pi rad                 Pi rad
        3       Forward Velocity          -Inf                    Inf
    Actions:
        Type: Box(2)
        Num   Action                      Min                     Max
        0     Acceleraton                 -Inf                    Inf
        1     Steering Rate               -Inf                    Inf
    Reward:
        None
    """

    def __init__(self, dt=0.02):
        """Init."""
        # pylint: disable=invalid-name
        self.dt = dt
        self.state = None

        action_high = np.array([np.finfo(np.float32).max,
                                np.finfo(np.float32).max], dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.pi,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        self.action_space = spaces.Box(-action_high,
                                       action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high,
                                            obs_high, dtype=np.float32)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Propagates car dynamics."""
        pos_x, pos_y, theta, vel = self.state
        u_theta, u_v = action
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        x_dot = vel * costheta
        y_dot = vel * sintheta
        theta_dot = vel * u_theta
        v_dot = u_v

        pos_x = pos_x + self.dt * x_dot
        pos_y = pos_y + self.dt * y_dot
        theta = theta + self.dt * theta_dot
        vel = vel + self.dt * v_dot

        theta = _angle_normalize(theta)
        self.state = (pos_x, pos_y, theta, vel)

        return np.array(self.state), 0, False, {}

    def reset(self):
        """Reset state to random value."""
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        print(self.state)


class ContCarEnv(NonNormContCarEnv):
    """Continuous car that outputs normalized observation and state in info."""

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)

    def step(self, action):
        """Propagate dynamics forward."""
        _, reward, done, info = super().step(action)
        info['state'] = np.array(self.state)

        return self._get_obs(), reward, done, info

    def reset(self):
        """Reset state to random value."""
        self.state = super().reset()
        return self._get_obs()

    def _get_obs(self):
        pos_x, pos_y, theta, vel = self.state
        return np.array([pos_x, pos_y, np.cos(theta), np.sin(theta), vel])


def _angle_normalize(ang):
    return ((ang+np.pi) % (2*np.pi)) - np.pi
