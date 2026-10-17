# setup.py
from setuptools import setup, find_packages

setup(
    name="basketworld",
    version="0.1.0",
    author="Your Name",
    description="A grid-based 3v3 basketball simulation for reinforcement learning research.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "matplotlib",
        "tqdm",
        "jax>=0.8.0",
        "flax>=0.12.6",
        "optax>=0.2.8",
        "orbax-checkpoint>=0.11.36",
        "pytest",
        "black",
        "isort"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
