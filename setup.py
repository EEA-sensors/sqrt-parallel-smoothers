import codecs
import os.path
import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""


# Get the package's version number of the __init__.py file
def read(rel_path):
    """Read the file located at the provided relative path."""
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    """Get the package's version number.
    We fetch the version  number from the `__version__` variable located in the
    package root's `__init__.py` file. This way there is only a single source
    of truth for the package's version number.
    """
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


INSTALL_REQS = read("requirements.txt").splitlines()

setuptools.setup(
    name="parsmooth",
    author="Adrien Corenflos",
    version=get_version("parsmooth/__init__.py"),
    python_requires='>=3.8,<=3.12',
    description="Parallel non-linear smoothing and parameter estimation for state space models",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQS,
    long_description_content_type="text/markdown",
    keywords="bayesian smoothing filtering parallel kalman sigma-points extended",
    license="MIT Licence",
)
