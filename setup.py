"""Setup file."""

from setuptools import setup

setup(
    name='gym-ccc',
    version='0.0.1',
    author='acxz',
    long_description='',
    description='Environments for continuous classical control problems with' +
    'dynamical state information',
    packages=['gym_ccc'],
    install_requires=[
        'gym',
        'gym-copter',
        'numpy',
    ],
)
