from gym.envs.registration import register

register(
    id='ContinuousPendulum-v0',
    entry_point='gym_continuous_classic_control.envs:ContinuousPendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NormalizedContinuousPendulum-v0',
    entry_point='gym_continuous_classic_control.envs:NormalizedContinuousPendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

from gym_continuous_classic_control.envs.continuous_pendulum import ContinuousPendulumEnv
from gym_continuous_classic_control.envs.continuous_pendulum import NormalizedContinuousPendulumEnv
