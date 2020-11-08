"""ContinuousPendulum."""
from gym import spaces
from gym.envs import classic_control

import numpy as np


# pylint: disable=too-many-instance-attributes
class PendulumNonNormEnv(classic_control.PendulumEnv):
    """Observation output is the non normalized state of the system."""

    # pylint: disable=too-many-arguments
    # pylint: disable=super-init-not-called
    def __init__(self, gravity=10.0, mass=1, length=1, damping=1, dt=0.05):
        """Init."""
        self.dt = dt
        self.g = gravity
        self.m = mass
        self.length = length
        self.damping = damping
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
        """Propagate the dynamics."""
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        length = self.length
        damping = self.damping
        dt = self.dt

        u = u[0]
        self.last_u = u  # for rendering

        newth = th + thdot * dt
        newth = angle_normalize(newth)
        newthdot = thdot + (-g / length * np.sin(th + np.pi) -
                            (damping * thdot + u) / (m * length ** 2)) * dt

        self.state = np.array([newth, newthdot])
        return self.state, 0, False, {}

    def reset(self):
        """Reset environment to random initial condition."""
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self.state


class PendulumEnv(PendulumNonNormEnv):
    """Observation: normalized state. Info contains the state."""

    def __init__(self, **kwargs):
        """Init according to super."""
        super().__init__(**kwargs)

    def step(self, u):
        """Add state to the info dict."""
        obs, reward, done, info = super().step(u)
        info['state'] = self.state

        return self._get_obs(), reward, done, info

    def reset(self):
        """Reset environment according to super class."""
        self.state = super().reset()
        return self._get_obs()


def angle_normalize(x):
    """Ensure angle is between -pi and pi."""
    return (((x+np.pi) % (2*np.pi)) - np.pi)
