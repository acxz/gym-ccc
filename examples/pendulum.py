"""Example pendulum."""
import gym

import numpy as np


def main():
    """Propagates pendulum dynamics."""

    def my_reset():
        init_state = np.array([np.pi, 0])
        return init_state

    kwargs = {'gravity': 10, 'mass': 1, 'length': 1, 'dt': 0.05,
              'custom_reset': my_reset}
    env = gym.make('gym_ccc.envs:PendulumCont-v0',
                   **kwargs)

    env.reset()
    while True:
        obs, reward, _, info = env.step(np.array([0]))
        print('time: ' + str(info['time']))
        print('reward: ' + str(reward))
        print('state: ' + str(info['state']))
        print('obs: ' + str(obs))
        print()
        env.render('human')


if __name__ == '__main__':
    main()
