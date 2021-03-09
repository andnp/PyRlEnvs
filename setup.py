from setuptools import setup, find_packages

setup(
    name='PyRlEnvs',
    url='https://github.com/andnp/PyRlEnvs.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*', 'scripts*']),
    install_requires=[
        "numpy>=1.19.5",
        "numba>=0.52.0",
        "RlGlue>=0.2",
    ],
    version=0.14,
    license='MIT',
    description='A handful of fast environments for running RL experiments',
    long_description='todo',
)
