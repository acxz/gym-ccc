"""Register environments."""
from gymnasium.envs.registration import register

from gym_ccc.envs.car import CarEnv  # noqa: F401
from gym_ccc.envs.car import CarNonNormEnv  # noqa: F401
from gym_ccc.envs.cartpole import CartPoleEnv  # noqa: F401
from gym_ccc.envs.cartpole import CartPoleNonNormEnv  # noqa: F401
# from gym_ccc.envs.multirotor import Multirotor2DSimpEnv  # noqa: F401, E501
from gym_ccc.envs.multirotor import Multirotor2DSimpNonNormEnv  # noqa: F401, E501
# from gym_ccc.envs.multirotor import MultirotorEnv  # noqa: F401
from gym_ccc.envs.multirotor import MultirotorNonNormEnv  # noqa: F401
# from gym_ccc.envs.multirotor import MultirotorSimpEnv  # noqa: F401
from gym_ccc.envs.multirotor import MultirotorSimpNonNormEnv  # noqa: F401
from gym_ccc.envs.pendulum import PendulumEnv  # noqa: F401
from gym_ccc.envs.pendulum import PendulumNonNormEnv  # noqa: F401

register(
    id='CarCont-v0',
    entry_point='gym_ccc.envs:CarEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='CarNonNormCont-v0',
    entry_point='gym_ccc.envs:CarNonNormEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='CartPoleCont-v0',
    entry_point='gym_ccc.envs:CartPoleEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='CartPoleNonNormCont-v0',
    entry_point='gym_ccc.envs:CartPoleNonNormEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='Multirotor2DSimpNonNormCont-v0',
    entry_point='gym_ccc.envs:Multirotor2DSimpNonNormEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='MultirotorNonNormCont-v0',
    entry_point='gym_ccc.envs:MultirotorNonNormEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='PendulumCont-v0',
    entry_point='gym_ccc.envs:PendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='PendulumNonNormCont-v0',
    entry_point='gym_ccc.envs:PendulumNonNormEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)
