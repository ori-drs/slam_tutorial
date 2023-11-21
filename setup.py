from distutils.core import setup
from setuptools import find_packages

setup(
    name="slam_tutorial",
    version="0.1.0",
    author="Matias Mattamala",
    author_email="matias@robots.ox.ac.uk",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    description="SLAM tutorial presented in the ORIentate Seminar",
)
