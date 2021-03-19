# Copyright (C) 2021 Vincent Russo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Information about qustop and dependencies."""
__all__ = ["about"]

import inspect
import platform
import sys

from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version

import toqito


PYTHON_VERSION = sys.version_info[0:3]


def about() -> None:
    """
    Displays information about qustop, core/optional packages, and
    Python version/platform information.
    """

    about_str = f"""
qustop: Quantum Optimizer: A Python toolkit for computing optimal values of various convex optimization problems in quantum information.
==============================================================================
Authored by: Vincent Russo, 2021 
Core Dependencies
-----------------
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}
Optional Dependencies
---------------------
Python Version:\t{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.{PYTHON_VERSION[2]}
Platform Info:\t{platform.system()} ({platform.machine()})"""
    print(about_str)


if __name__ == "__main__":
    about()