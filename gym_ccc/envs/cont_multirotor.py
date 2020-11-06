"""Continuous Multirotor Environment."""
import gym
from gym import spaces

import numpy as np

from pyquaternion import Quaternion


class NonNormContMultirotorEnv(gym.Env):
    """
    Continuous multirotor that outputs the state as the observation.

    Description:
        Multirotor dynamics where the control input are the moments and the
        thrust acting on the multirotor.
    Observation:
        Type: Box(4)
        Num     Observation                 Min                     Max
        0       x position                  -Inf                    Inf
        1       y position                  -Inf                    Inf
        2       z position                  -Inf                    Inf
        3       x velocity                  -Inf                    Inf
        4       y velocity                  -Inf                    Inf
        5       z velocity                  -Inf                    Inf
        6       w quaternion                0                       1
        7       x quaternion                0                       1
        8       y quaternion                0                       1
        9       z quaternion                0                       1
        10      (roll) x angular velocity   -Inf                    Inf
        11      (pitch) y angular velocity  -Inf                    Inf
        12      (yaw) z angular velocity    -Inf                    Inf
    Actions:
        Type: Box(1)
        Num   Action                        Min                     Max
        0     x moment                      -Inf                    Inf
        1     y moment                      -Inf                    Inf
        2     z moment                      -Inf                    Inf
        3     thrust                        -Inf                    Inf
    Reward:
        None
    Starting State:
        Hovering at a height of 1 meter.
    """

    def __init__(self, gravity=9.8, mass=1.0, inertia=np.diag(np.ones(3)),
                 dt=0.02):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        self.inertia = inertia
        # pylint: disable=invalid-name
        self.dt = dt
        self.state = np.zeros(13)

        action_high = np.array([np.finfo(np.float32).max,
                                np.finfo(np.float32).max,
                                np.finfo(np.float32).max,
                                np.finfo(np.float32).max],
                               dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             1,
                             1,
                             1,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        obs_low = np.array([np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            0,
                            0,
                            0,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min],
                           dtype=np.float32)

        self.action_space = spaces.Box(-action_high,
                                       action_high, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low,
                                            obs_high, dtype=np.float32)

    def step(self, action):
        """Propagate dynamics."""
        pos = self.state[0:3]
        vel = self.state[3:6]
        quat = self.state[6:10]
        ang_vel = self.state[10:13]

        moments = action[0:3]
        thrust = action[4]

        quat = Quaternion(quat)
        rotation_matrix = quat.rotation_matrix

        pos_dot = vel
        vel_dot = (thrust / self.mass) * \
            np.array([0, 0, 1]) @ rotation_matrix - \
            np.array([0, 0, 1]) * self.gravity
        quat_dot = quat.derivative(ang_vel)
        ang_vel_dot = np.inverse(self.inertia) @ \
            (moments - np.cross(ang_vel, self.inertia @ ang_vel))

        pos = pos + self.dt * pos_dot
        vel = vel + self.dt * vel_dot
        quat = quat + self.dt * quat_dot
        quat = quat.normalize
        ang_vel = ang_vel + self.dt * ang_vel_dot

        self.state = np.hcat(pos, vel, quat, ang_vel)

        return self.state, 0, False, {}

    def reset(self):
        """Reset to attitude stable state."""
        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        # Do an euler angle conversion for ease of readability
        print(self.state)
