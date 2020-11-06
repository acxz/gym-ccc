"""Register environments."""
from gym.envs.registration import register

import gym_continuous_classic_control.envs.continuous_car.ContinuousCarEnv
import gym_continuous_classic_control.envs.continuous_car. \
        NonNormalizedContinuousCarEnv
import gym_continuous_classic_control.envs.continuous_cartpole. \
        ContinuousCartPoleEnv
import gym_continuous_classic_control.envs.continuous_cartpole. \
        NonNormalizedContinuousCartPoleEnv
import gym_continuous_classic_control.envs.continuous_pendulum. \
        ContinuousPendulumEnv
import gym_continuous_classic_control.envs.continuous_pendulum. \
        NonNormalizedContinuousPendulumEnv

register(
    id='NonNormalizedContinuousPendulum-v0',
    entry_point=('gym_continuous_classic_control.envs:',
                 'NonNormalizedContinuousPendulumEnv'),
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='ContinuousPendulum-v0',
    entry_point='gym_continuous_classic_control.envs:ContinuousPendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NonNormalizedContinuousCartPole-v0',
    entry_point=('gym_continuous_classic_control.envs:',
                 'NonNormalizedContinuousCartPoleEnv'),
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='ContinuousCartPole-v0',
    entry_point='gym_continuous_classic_control.envs:ContinuousCartPoleEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NonNormalizedContinuousCar-v0',
    entry_point=('gym_continuous_classic_control.envs:',
                 'NonNormalizedContinuousCarEnv'),
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='ContinuousCar-v0',
    entry_point='gym_continuous_classic_control.envs:ContinuousCarEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)
