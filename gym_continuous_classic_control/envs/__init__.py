from gym.envs.registration import register

register(
    id='NonNormalizedContinuousPendulum-v0',
    entry_point='gym_continuous_classic_control.envs:NonNormalizedContinuousPendulumEnv',
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

from gym_continuous_classic_control.envs.continuous_pendulum import NonNormalizedContinuousPendulumEnv
from gym_continuous_classic_control.envs.continuous_pendulum import ContinuousPendulumEnv
