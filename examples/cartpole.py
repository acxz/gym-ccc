"""Example file for cartpole."""
import gym


def main():
    """Propagates cartpole dynamics."""
    kwargs = {'gravity': 10}
    env = gym.make('gym_ccc.envs:CartPoleCont-v0',
                   **kwargs)

    env.reset()
    while True:
        _, _, _, info = env.step(0)
        print('\r' + str(info['state']), end='')
        env.render('human')


if __name__ == '__main__':
    main()
