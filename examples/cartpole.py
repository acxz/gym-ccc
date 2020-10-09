import gym
import numpy as np

def main():
    env = gym.make('gym_continuous_classic_control.envs:ContinuousCartPole-v0')

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(0)
        print("\r" + str(info['state']), end="")
        env.render('human')

if __name__ == "__main__":
    main()
