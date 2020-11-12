"""ContinuousPendulum."""
from gym import spaces
from gym.envs import classic_control

import numpy as np


# pylint: disable=too-many-instance-attributes
class PendulumNonNormEnv(classic_control.PendulumEnv):
    """Observation output is the non normalized state of the system."""

    # pylint: disable=too-many-arguments
    # pylint: disable=super-init-not-called
    def __init__(self, gravity=10.0, mass=1, length=1, damping=0.1, dt=0.05,
                 max_speed=8, max_torque=2, custom_reset=None):
        """Init."""
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.dt = dt
        self.gravity = gravity
        self.mass = mass
        self.length = length
        self.damping = damping

        self.state = None
        self.time = 0

        self.custom_reset = custom_reset

        self.last_u = None
        self.viewer = None

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.seed()

    def step(self, u):
        """Propagate the dynamics."""
        theta, theta_dot = self.state

        gravity = self.gravity
        mass = self.mass
        length = self.length
        damping = self.damping
        # pylint: disable=invalid-name
        dt = self.dt
        self.time += dt

        torque = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = torque  # for rendering
        costs = self.angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + \
            0.001 * torque ** 2

        new_theta = theta + theta_dot * dt
        new_theta = self.angle_normalize(new_theta)
        new_theta_dot = theta_dot + \
            (-gravity / length * np.sin(theta + np.pi) -
             (damping * theta_dot + torque) /
             (mass * length ** 2)) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)

        self.state = np.array([new_theta, new_theta_dot])
        return self.state, -costs, False, {'time': self.time}

    def reset(self):
        """Reset environment."""
        self.time = 0
        if self.custom_reset is not None:
            self.state = self.custom_reset()
        else:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self.state, 0, False, {'time': self.time}

    @staticmethod
    def angle_normalize(angle):
        """Ensure angle is between -pi and pi."""
        return ((angle+np.pi) % (2*np.pi)) - np.pi


class PendulumEnv(PendulumNonNormEnv):
    """Observation: normalized state. Info contains the state."""

    def __init__(self, **kwargs):
        """Init according to super."""
        super().__init__(**kwargs)

        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -self.max_speed], dtype=np.float32),
            high=np.array([1, 1, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, u):
        """Add state to the info dict."""
        _, reward, done, info = super().step(u)
        info['state'] = self.state

        return self._get_obs(), reward, done, info

    def reset(self):
        """Reset environment according to super class."""
        _, reward, done, info = super().reset()
        info['state'] = self.state
        return self._get_obs(), reward, done, info
