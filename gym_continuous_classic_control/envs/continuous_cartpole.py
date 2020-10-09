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
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
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
    Episode Termination:
        Episode length is greater than 1000, unless specified.
    """

    def __init__(self):
        super().__init__()

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

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        theta = angle_normalize(theta)
        self.state = (x, x_dot, theta, theta_dot)

        return np.array(self.state), 0, False, {}

class ContinuousCartPoleEnv(NonNormalizedContinuousCartPoleEnv):
    def __init__(self):
        super().__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['state'] = np.array(self.state)

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
