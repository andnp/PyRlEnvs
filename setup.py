from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

print(find_packages(exclude=['tests*', 'scripts*']))

setup(
    name='PyRlEnvs',
    url='https://github.com/andnp/PyRlEnvs.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*', 'scripts*']),
    package_data={"PyRlEnvs": ["py.typed"]},
    version='1.0.0',
    license='MIT',
    description='A handful of fast environments for running RL experiments',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "numba>=0.53.0",
        "RlGlue-andnp>=0.3",
        "scipy>=1.5.0",
    ],
    extras_require={},
)
