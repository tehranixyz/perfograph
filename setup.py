from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="perfograph",
    version="0.1",
    author="Ali Tehrani",
    description="A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tehranixyz/perfograph",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=required,
    python_requires=">=3.7,<=3.10.13",
)
