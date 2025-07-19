# setup.py
from setuptools import setup, find_packages

setup(
    name="basketworld",
    version="0.1.0",
    author="Your Name",
    description="A grid-based 3v3 basketball simulation for reinforcement learning research.",
    packages=find_packages(),
    install_requires=[
        "gymnasium==0.29.1",
        "numpy>=1.24",
        "matplotlib",
        "tqdm",
        "torch>=2.0",
        "torchvision",
        "stable-baselines3==2.2.1",
        "pytest",
        "black",
        "isort"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)