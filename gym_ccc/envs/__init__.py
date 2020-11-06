"""Register environments."""
from gym.envs.registration import register

from gym_ccc.envs.cont_car import ContCarEnv  # noqa: F401
from gym_ccc.envs.cont_car import NonNormContCarEnv  # noqa: F401
from gym_ccc.envs.cont_cartpole import ContCartPoleEnv  # noqa: F401
from gym_ccc.envs.cont_cartpole import NonNormContCartPoleEnv  # noqa: F401
from gym_ccc.envs.cont_multirotor import NonNormContMultirotorEnv  # noqa: F401
from gym_ccc.envs.cont_pendulum import ContPendulumEnv  # noqa: F401
from gym_ccc.envs.cont_pendulum import NonNormContPendulumEnv  # noqa: F401

register(
    id='ContCar-v0',
    entry_point='gym_ccc.envs:ContCarEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NonNormContCar-v0',
    entry_point='gym_ccc.envs:NonNormContCarEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)


register(
    id='ContCartPole-v0',
    entry_point='gym_ccc.envs:ContCartPoleEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NonNormContCartPole-v0',
    entry_point='gym_ccc.envs:NonNormContCartPoleEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)


register(
    id='ContPendulum-v0',
    entry_point='gym_ccc.envs:ContPendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)

register(
    id='NonNormContPendulum-v0',
    entry_point='gym_ccc.envs:NonNormContPendulumEnv',
    reward_threshold=None,
    nondeterministic=False,
    max_episode_steps=None,
    kwargs={},
)
