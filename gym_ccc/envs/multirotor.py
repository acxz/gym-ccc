"""Continuous Multirotor Environment."""
import sys

import warnings

import gym
from gym import spaces

try:
    import gym_copter.rendering.twod
except ImportError:
    warnings.warn('multirotor gui rendering is disabled, defaulting to console \
printing')

import numpy as np

from pyquaternion import Quaternion


# pylint: disable=too-many-instance-attributes
class MultirotorNonNormEnv(gym.Env):
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
        Quadratic cost to stable hover at pos = [5, 5, 5]
    Starting State:
        Height of 0 meter with a heading of 0 at stable attitude.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, gravity=9.8, mass=1.0, inertia=np.diag(np.ones(3)),
                 dt=0.02, custom_reset=None):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        self.inertia = inertia
        # pylint: disable=invalid-name
        self.dt = dt

        self.state = None
        self.time = 0

        self.custom_reset = custom_reset

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

        goal_state = np.array([5, 5, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        cost = np.transpose(self.state - goal_state) @ \
            np.eye(self.state.shape[0]) @ (self.state - goal_state)

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
        self.time += self.dt

        self.state = np.hstack((pos, vel, quat, ang_vel))

        return self.state, -cost, False, {'time': self.time}

    def reset(self):
        """Reset environment."""
        self.time = 0
        if self.custom_reset is not None:
            self.state = self.custom_reset()
        else:
            self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        print(self.state)


class MultirotorSimpNonNormEnv(gym.Env):
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
        Quadratic cost to stable hover at pos = [5, 5, 5]
    Starting State:
        Height of 0 meter with a heading of 0 at stable attitude.
    """

    def __init__(self, gravity=9.8, mass=1.0, dt=0.02, custom_reset=None):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        # pylint: disable=invalid-name
        self.dt = dt

        self.state = None
        self.time = 0

        self.custom_reset = custom_reset

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

        goal_state = np.array([5, 5, 5, 0, 0, 0, 1, 0, 0, 0])
        cost = np.transpose(self.state - goal_state) @ \
            np.eye(self.state.shape[0]) @ (self.state - goal_state)

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
        self.time += self.dt

        self.state = np.hstack((pos, vel, quat))

        return self.state, -cost, False, {'time': self.time}

    def reset(self):
        """Reset to attitude stable state."""
        self.time = 0
        if self.custom_reset is not None:
            self.state = self.custom_reset()
        else:
            self.state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        print(self.state)


class Multirotor2DSimpNonNormEnv(gym.Env):
    """
    Continuous multirotor on the y - z axis with higher level control.

    Description:
        Multirotor dynamics where the control input is the x (roll)
        angular velocity and the thrust acting on the multirotor.
    Observation:
        Type: Box(4)
        Num     Observation                 Min                     Max
        0       y position                  -Inf                    Inf
        1       z position                  -Inf                    Inf
        2       y velocity                  -Inf                    Inf
        3       z velocity                  -Inf                    Inf
        4       roll angle                  -Pi                     Pi
    Actions:
        Type: Box(1)
        Num   Action                        Min                     Max
        0     (roll) x angular velocity     -Inf                    Inf
        1     thrust                        -Inf                    Inf
    Reward:
        Quadratic cost to stable hover at pos = [5, 5]
    Starting State:
        Height of 0 meter with a heading of 0 at stable attitude.
    """

    frames_per_second = 50
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': frames_per_second
    }

    def __init__(self, gravity=9.8, mass=1.0, dt=0.02, custom_reset=None):
        """Init quadrotor env."""
        super().__init__()

        self.gravity = gravity
        self.mass = mass
        # pylint: disable=invalid-name
        self.dt = dt

        self.state = None
        self.time = 0

        self.custom_reset = custom_reset

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

        # Support for rendering
        if 'gym_copter.rendering.twod' in sys.modules:
            self.renderer = None
        else:
            self.renderer = False
        self.pose = None

    def step(self, action):
        """Propagate dynamics."""
        pos = self.state[0:2]
        vel = self.state[2:4]
        roll_ang = self.state[4]

        roll_ang_vel = action[0]
        thrust = action[1]

        goal_state = np.array([5, 5, 0, 0, 0])
        cost = np.transpose(self.state - goal_state) @ \
            np.eye(self.state.shape[0]) @ (self.state - goal_state)

        pos_dot = vel
        rotation_matrix = np.array([[np.cos(roll_ang), -np.sin(roll_ang)],
                                    [np.sin(roll_ang), np.cos(roll_ang)]])
        vel_dot = (thrust / self.mass) * \
            np.array([0, 1]) @ rotation_matrix - \
            np.array([0, 1]) * self.gravity
        roll_ang_dot = roll_ang_vel

        pos = pos + self.dt * pos_dot
        vel = vel + self.dt * vel_dot
        roll_ang = roll_ang + self.dt * roll_ang_dot
        roll_ang = self.angle_normalize(roll_ang)
        self.time += self.dt

        self.state = np.hstack((pos, vel, roll_ang))

        # Support for rendering
        self.pose = -pos[0], -pos[1], roll_ang

        return self.state, -cost, False, {'time': self.time}

    def reset(self):
        """Reset to attitude stable state."""
        self.time = 0
        if self.custom_reset is not None:
            self.state = self.custom_reset()
        else:
            self.state = np.array([0, 0, 0, 0, 0])
        return self.state

    def render(self, mode='human'):
        """Show the current state."""
        # Print out state instead of cool gui if renderer not available
        if self.renderer is False:
            print(self.state)
            return self.state

        # Creater renderer
        if self.renderer is None:
            self.renderer = gym_copter.rendering.twod.TwoDRenderer()

        # Just a flag to show the props spinning
        flight_status = True

        self.renderer.render(self.pose, flight_status)
        return self.renderer.complete(mode)

    @staticmethod
    def angle_normalize(ang):
        """Normalize angle between -pi and pi."""
        return ((ang+np.pi) % (2*np.pi)) - np.pi
