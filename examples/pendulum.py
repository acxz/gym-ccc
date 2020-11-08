"""Example pendulum."""
import gym

import numpy as np


def main():
    """Propagates pendulum dynamics."""
    kwargs = {'gravity': 9.8, 'mass': 1, 'length': 1, 'damping': 1, 'dt': 0.05}
    env = gym.make('gym_ccc.envs:PendulumCont-v0',
                   **kwargs)

    env.reset()
    while True:
        _, _, _, info = env.step(np.array([0]))
        print('\r' + str(info['state']), end='')
        env.render('human')


if __name__ == '__main__':
    main()
