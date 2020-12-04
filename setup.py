"""Setup file."""

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='gym-ccc',
    version='0.0.1',
    author='acxz',
    long_description=long_description,
    description='Environments for continuous classical control problems with' +
    'dynamical state information',
    packages=setuptools.find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'pyquaternion',
    ],
    extra_requires={
        'copter-render': ['gym-copter'],
    }
)
