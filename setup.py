"""Setup file."""

from setuptools import setup

setup(
    name='gym-continuous-classic-control',
    version='0.0.1',
    author='acxz',
    long_description='',
    description='Environments for classical control problems with dynamical' +
            'state information',
    packages=['gym_continuous_classic_control'],
    install_requires=[
        'gym',
        'numpy',
    ],
)
