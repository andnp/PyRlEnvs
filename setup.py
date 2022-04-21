from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='PyRlEnvs-andnp',
    url='https://github.com/andnp/PyRlEnvs.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*', 'scripts*']),
    package_data={"PyRlEnvs": ["py.typed"]},
    version="1.1.0",
    license="MIT",
    description="A handful of fast environments for running RL experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["numpy>=1.19.5", "numba>=0.52.0", "scipy>=1.5.4", "rlglue-andnp"],
    extras_require={
        "dev": [
            "build",
            "commitizen",
            "mypy>=0.942",
            "flake8>=4.0.1",
            "gym>=0.18.0",
            "pre-commit",
            "pygame",
            "pipenv-setup[black]",
            "twine",
        ],
    },
)
