"""Example cartpole."""
import gym

import numpy as np


def main():
    """Propagates cartpole dynamics."""
    env = gym.make('gym_ccc.envs:CartPoleCont-v0')

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
