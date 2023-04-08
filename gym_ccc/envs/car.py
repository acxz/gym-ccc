"""Continuous Car Env."""
import gymnasium as gym
from gymnasium import spaces

import numpy as np


class CarNonNormEnv(gym.Env):
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
        Quadratic cost on reaching [5, 5, 0, 0]
    """

    def __init__(self, dt=0.02, custom_reset=None):
        """Init."""
        # pylint: disable=invalid-name
        self.dt = dt

        self.state = None
        self.time = 0

        self.custom_reset = custom_reset

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

    def step(self, action):
        """Propagates car dynamics."""
        pos_x, pos_y, theta, vel = self.state
        u_theta, u_v = action
        self.time += self.dt

        cost = (pos_x - 5) ** 2 + (pos_y - 5) ** 2 + theta ** 2 + vel ** 2

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

        theta = self.angle_normalize(theta)
        self.state = np.array([pos_x, pos_y, theta, vel])

        return self.state, -cost, False, {'time': self.time}

    def reset(self, seed=None, options=None):
        """Reset environment."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.time = 0
        if self.custom_reset is not None:
            self.state = self.custom_reset()
        else:
            self.state = \
                self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    @staticmethod
    def angle_normalize(ang):
        """Normalize angle between -pi and pi."""
        return ((ang+np.pi) % (2*np.pi)) - np.pi

    def render(self, mode='human'):
        """Show the current state."""
        print(self.state)


class CarEnv(CarNonNormEnv):
    """Continuous car that outputs normalized observation and state in info."""

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             1,
                             1,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        self.observation_space = spaces.Box(-obs_high,
                                            obs_high, dtype=np.float32)

    def step(self, action):
        """Propagate dynamics forward."""
        _, reward, done, info = super().step(action)
        info['state'] = np.array(self.state)

        return self._get_obs(), reward, done, info

    def reset(self, seed=None, options=None):
        """Reset state to random value."""
        self.state = super().reset(seed=seed, options=options)
        return self._get_obs()

    def _get_obs(self):
        pos_x, pos_y, theta, vel = self.state
        return np.array([pos_x, pos_y, np.cos(theta), np.sin(theta), vel])
