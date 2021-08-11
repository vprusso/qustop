Installing
==========

The :code:`qustop` package can be installed from PyPi via

.. code-block:: bash

    pip install qustop

To build from source, you may also run the following command from the
top-level package directory.

.. code-block:: bash

    python setup.py install

To test installation, run

.. code-block:: python

    >>> import qustop
    >>> qustop.about()

    qustop: Quantum Optimizer: A Python toolkit for computing optimal values of various convex
    optimization problems in quantum information.
    ==============================================================================
    Authored by: Vincent Russo, 2021

    Core Dependencies
    -----------------
    CVXPY Version:  1.1.13
    NumPy Version:  1.21.0
    SciPy Version:  1.6.3
    Optional Dependencies
    ---------------------
    Python Version: 3.9.2
    Platform Info:  Darwin (x86_64)


This prints out version information about core requirements and optional conic optimization software packages that
:code:`qustop` can interface with.

Testing
=======

The :code:`pytest` module is used for testing. In order to run and :code:`pytest`, you will need to ensure it is
installed on your machine. Consult the `pytest <https://docs.pytest.org/en/latest/>`_ website for more information. To
run the suite of tests for :code:`qustop`, run the following command in the root directory of this project:

.. code-block:: bash

    pytest --cov-report term-missing --cov=qustop tests/

Contributing
============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing guide <https://github.com/vprusso/qustop/blob/master/.github/CONTRIBUTING.md>`_.

Citing
======

You can cite :code:`qustop` using the following DOI: XXX.

If you are using the :code:`qustop` software package in research work, please
include an explicit mention of :code:`qustop` in your publication. Something
along the lines of:

    To solve problem "X" we used `qustop`; a package for studying quantum state
    optimization scenarios.

A BibTeX entry that you can use to cite :code:`qustop` is provided here:

.. code-block:: bash

    @misc{qustop,
       author       = {Vincent Russo},
       title        = {qustop: A {P}ython package for investigating quantum state optimization, version 0.1},
       howpublished = {\url{https://github.com/vprusso/qustop}},
       month        = May,
       year         = 2021,
       doi          = {XXX}
     }
