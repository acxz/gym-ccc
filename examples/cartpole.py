"""Example cartpole."""
import gymnasium as gym

import numpy as np


def main():
    """Propagates cartpole dynamics."""
    env = gym.make('gym_ccc.envs:CartPoleCont-v0', render_mode="human")

    env.reset()
    while True:
        action = 0
        obs, reward, _, _, info = env.step(action)
        print('time: ' + str(info['time']))
        print('reward: ' + str(reward))
        print('state: ' + str(info['state']))
        print('obs: ' + str(obs))
        print()
    env.close()


if __name__ == '__main__':
    main()
