import gym
import numpy as np

def main():
    kwargs = {'g': 9.8, 'm': 1, 'l': 1, 'b': 1, 'dt': 0.05}
    env = gym.make('gym_continuous_classic_control.envs:ContinuousPendulum-v0',
            **kwargs)

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(np.array([0]))
        print("\r" + str(info['state']), end="")
        env.render('human')

if __name__ == "__main__":
    main()
