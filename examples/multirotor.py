"""Example file for 2D Multirotor."""
import gymnasium as gym

import numpy as np


# pylint: disable=too-many-locals
def control(state):
    """Compute control given state."""
    gravity = 9.8
    mass = 1

    pos_y_goal = -5
    pos_z_goal = 5
    vel_y_goal = 0
    vel_z_goal = 0
    # roll_ang_goal = 0 implied

    pos_y = state[0]
    pos_z = state[1]
    vel_y = state[2]
    vel_z = state[3]
    roll_ang = state[4]

    kp_thrust = 0.005
    kd_thrust = 0.1
    kp_roll_ang_target = 0.1
    kd_roll_ang_target = 2.3
    kp_roll_ang_vel = 1

    thrust_feedforward = gravity * mass
    pos_z_error = pos_z_goal - pos_z
    vel_z_error = vel_z_goal - vel_z
    thrust_feedback = kp_thrust * pos_z_error + kd_thrust * vel_z_error
    thrust = thrust_feedforward + thrust_feedback

    pos_y_error = pos_y_goal - pos_y
    vel_y_error = vel_y_goal - vel_y
    roll_ang_target = kp_roll_ang_target * pos_y_error \
        + kd_roll_ang_target * vel_y_error
    roll_ang_error = roll_ang_target - roll_ang

    roll_ang_vel = kp_roll_ang_vel * roll_ang_error
    return np.array([roll_ang_vel, thrust])


def main():
    """Propagates multirotor dynamics."""
    env = gym.make('gym_ccc.envs:Multirotor2DSimpNonNormCont-v0')

    obs = env.reset()
    while True:
        action = control(obs)
        # print('\r' + str(action), end='')
        obs, _, _, _ = env.step(action)
        # print('\r' + str(info['state']), end='')
        print('\r' + str(obs), end='')
        env.render('human')


if __name__ == '__main__':
    main()
