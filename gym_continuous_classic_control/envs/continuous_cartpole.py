import math
from gym import spaces
from gym.envs import classic_control
import numpy as np


class NonNormalizedContinuousCartPoleEnv(classic_control.CartPoleEnv):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -Inf                    Inf
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -Pi rad (-180 deg)      Pi rad (180 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Box(1)
        Num   Action                      Min                     Max
        0     Force on the cart           -Inf                    Inf
    Reward:
        None
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    """

    def __init__(self, gravity=9.8, masscart=1.0, masspole=0.1, polelength=1.0, tau=0.02):
        super().__init__()

        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.polelength = polelength
        self.length = self.polelength/2 # For rendering purposes
        self.tau = tau

        action_high = np.array([np.finfo(np.float32).max], dtype=np.float32)

        obs_high = np.array([np.finfo(np.float32).max,
                             np.finfo(np.float32).max,
                             np.pi,
                             np.finfo(np.float32).max],
                            dtype=np.float32)

        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = self.masscart + self.masspole * sintheta ** 2
        xacc = 1/temp * (self.masspole * sintheta * (self.polelength * theta_dot**2 \
            + self.gravity * costheta) + force)
        thetaacc = 1/(self.polelength * temp) * (- self.masspole * self.polelength * \
                theta_dot**2 * costheta * sintheta - (self.masscart + \
                    self.masspole) * self.gravity * sintheta - force * costheta)

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        theta = angle_normalize(theta)
        self.state = (x, x_dot, theta, theta_dot)

        return np.array(self.state), 0, False, {}

class ContinuousCartPoleEnv(NonNormalizedContinuousCartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['state'] = np.array(self.state)

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
