# gym-ccc
A repo of gym environments for continuous classic control (ccc) problems. In
particular, this repo offers gym environments where the observation is the state
of the system as given by the dynamics. In addition to these environments,
normalized environments are provided which contain the real state in the `info`
output and the observation is instead some normalized version of the state.

The main highlights are:
1) non normalized observation corresponding directly to the dynamical state
2) normalized observation with dynamical state captured in `info['state']`
3) action spaces are continuous
4) system parameters (mass, length, etc.) can be specificed
5) reset function (to specify initial conditions) can be specified.

The motivation for this is so that gym environments can be used for control
problems where the states/observations are not traditionally normalized. The
observations in gym environments are normalized since the observation vector is
directly passed into neural network models and dealing with normalization within
the neural network for each environment is not feasible. However, this also
means that gym environments can't be used for control methods that required the
non-normalized states/observations.

The normalized environments are directly extended from the non-normalized
environments to ensure the same dynamics are used.

If you want to use a different kind of normalization then feel free to extend
the original gym environments and output a normalized observation of your own
choosing. Similarly, you can extend the gym environments to disregard the given
reward shape and implement your own. See gym wrappers for how to do this.
(https://alexandervandekleut.github.io/gym-wrappers/)


## Installation

You'll have to install a dependency not available on PyPi manually first.

- [gym-copter](https://github.com/simondlevy/gym-copter)

The rest of the dependencies will come through with the `pip install` command.

```bash
pip install .
```

## Usage

As you would any other gym environment. See the examples directory for some
example code of creating and stepping through the gym env.
