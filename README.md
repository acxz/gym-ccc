# gym-continuous-classic-control
A repo of gym environments for classic control problems. In particular, this
repo offers gym environments where the observation is the state of the
system as given by the dynamics. In addition to these environments, normalized
environments are provided which contain the real state in the `infos` output and
the observation is instead some normalized version of the state.

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
reward shape and implement your own.

The main purpose of this library is to provide premade gym environments
implementing dynamics of traditional control problems that incorporate the
non-normalized states of the system, whether that is through not normalizing the
observation output of `step` or by adding the state information in the output
`info` dict of `step`.

## Installation

```bash
pip install gym-continuous-classic-control
```

## Usage

TODO
