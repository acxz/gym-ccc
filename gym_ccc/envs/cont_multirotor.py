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
        Height of 0 meter with a heading of 0 at stable attitude.
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


class NonNormContMultirotorSimplifiedEnv(gym.Env):
    """
    Continuous multirotor with higher level control.

    Description:
        Multirotor dynamics where the control input is the angular rates and
        the thrust acting on the multirotor.
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
    Actions:
        Type: Box(1)
        Num   Action                        Min                     Max
        0     (roll) x angular velocity     -Inf                    Inf
        1     (pitch) y angular velocity    -Inf                    Inf
        2     (yaw) z angular velocity      -Inf                    Inf
        3     thrust                        -Inf                    Inf
    Reward:
        None
    Starting State:
        Height of 0 meter with a heading of 0 at stable attitude.
    """

    def __init__(self, gravity=9.8, mass=1.0, dt=0.02):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        # pylint: disable=invalid-name
        self.dt = dt
        self.state = np.zeros(10)

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
                             1],
                            dtype=np.float32)

        obs_low = np.array([np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            np.finfo(np.float32).min,
                            0,
                            0,
                            0],
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

        ang_vel = action[0:3]
        thrust = action[4]

        quat = Quaternion(quat)
        rotation_matrix = quat.rotation_matrix

        pos_dot = vel
        vel_dot = (thrust / self.mass) * \
            np.array([0, 0, 1]) @ rotation_matrix - \
            np.array([0, 0, 1]) * self.gravity
        quat_dot = quat.derivative(ang_vel)

        pos = pos + self.dt * pos_dot
        vel = vel + self.dt * vel_dot
        quat = quat + self.dt * quat_dot
        quat = quat.normalize

        self.state = np.hcat(pos, vel, quat)

        return self.state, 0, False, {}

    def reset(self):
        """Reset to attitude stable state."""
        self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        # Do an euler angle conversion for ease of readability
        print(self.state)


class NonNormContMultirotor2DSimplifiedEnv(gym.Env):
    """
    Continuous multirotor on the x - z axis with higher level control.

    Description:
        Multirotor dynamics where the control input is the y (pitch)
        angular velocity and the thrust acting on the multirotor.
    Observation:
        Type: Box(4)
        Num     Observation                 Min                     Max
        0       x position                  -Inf                    Inf
        1       z position                  -Inf                    Inf
        2       x velocity                  -Inf                    Inf
        3       z velocity                  -Inf                    Inf
        4       pitch angle                 -Pi                     Pi
    Actions:
        Type: Box(1)
        Num   Action                        Min                     Max
        0     (pitch) y angular velocity    -Inf                    Inf
        1     thrust                        -Inf                    Inf
    Reward:
        None
    Starting State:
        Height of 0 meter with a heading of 0 at stable attitude.
    """

    def __init__(self, gravity=9.8, mass=1.0, dt=0.02):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        # pylint: disable=invalid-name
        self.dt = dt
        self.state = np.zeros(5)

        action_high = np.array([np.finfo(np.float32).max,
                                np.finfo(np.float32).max],
                               dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.pi],
                            dtype=np.float32)

        self.action_space = spaces.Box(-action_high,
                                       action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high,
                                            obs_high, dtype=np.float32)

    def step(self, action):
        """Propagate dynamics."""
        pos = self.state[0:2]
        vel = self.state[2:4]
        pitch_ang = self.state[4]

        pitch_ang_vel = action[0]
        thrust = action[1]

        pos_dot = vel
        rotation_matrix = np.array([[np.cos(pitch_ang), -np.sin(pitch_ang)],
                                    [np.sin(pitch_ang), np.cos(pitch_ang)]])
        vel_dot = (thrust / self.mass) * \
            np.array([0, 1]) @ rotation_matrix - \
            np.array([0, 1]) * self.gravity
        pitch_ang_dot = pitch_ang_vel

        pos = pos + self.dt * pos_dot
        vel = vel + self.dt * vel_dot
        pitch_ang = pitch_ang + self.dt * pitch_ang_dot
        pitch_ang = _angle_normalize(pitch_ang)

        self.state = np.hcat(pos, vel, pitch_ang)

        return self.state, 0, False, {}

    def reset(self):
        """Reset to attitude stable state."""
        self.state = np.array([0, 0, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        print(self.state)


def _angle_normalize(ang):
    """Normalize angle between -pi and pi."""
    return ((ang+np.pi) % (2*np.pi)) - np.pi
