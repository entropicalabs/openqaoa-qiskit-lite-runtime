from setuptools import setup, find_namespace_packages
from os import getcwd

current_path = getcwd()

with open("README.md", "r") as desc:
    long_description = desc.read()

requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

setup(
    name="openqaoa-qiskit-runtime",
    python_requires=">=3.10, <3.11",
    license="MIT",
    long_description=long_description,
    install_requires=requirements
)
