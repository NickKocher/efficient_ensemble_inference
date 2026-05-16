import os

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

setup(
    name="raaml",
    author="xxxxxxx",
    author_email="xxxx@xxx.xxxxxx",
    description="Automated machine learning.",
    long_description=None,
    long_description_content_type="text/markdown",
    version=0.1,
    packages=find_packages(exclude=["test", "scripts", "examples"]),
    extras_require=None,
    install_requires=[],
    include_package_data=True,
    license="BSD3",
    platforms=["Linux"],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires="==3.11.5",
)
