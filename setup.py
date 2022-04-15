import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "cvx",
    "cvxpy",
    "numpy",
    "picos",
    "scipy",
    "scikit-image",
    "toqito",
]

setuptools.setup(
    name="qustop",
    version="0.0.2",
    author="Vincent Russo",
    author_email="vincentrusso1@gmail.com",
    description="Python toolkit for studying quantum state optimization problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vprusso/toqito",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    test_suite="tests",
)
