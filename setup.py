import os
from setuptools import setup, find_packages

setup(
    name="distml",
    version=os.environ.get("VERSION"),
    author="The DistML Authors",
    author_email="",
    description=
    "DistML is a runtime libraray for distributed deep learning training.",
    long_description=
    """DistML is a Ray extension library to support large-scale distributed ML training on heterogeneous 
       multi-node multi-GPU clusters.""",
    url="https://github.com/ray-project/distml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    install_requires=[
        "ray",
        "pandas",
        "tabulate",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pydocstyle",
            "prospector",
        ]
    },
    packages=find_packages(),
    python_requires=">=3.6",
)
