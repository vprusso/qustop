Getting started
===============

Installing
^^^^^^^^^^

1. Ensure you have Python 3.8 or greater installed on your machine.

2. Consider using a `virtual environment <https://packaging.python.org/guides/installing-using-pip-and-virtualenv/>`_.

3. The preferred way to install the :code:`qustop` package is via :code:`pip`.

.. code-block:: bash

    pip install qustop

Alternatively, to install, you may also run the following command from the
top-level package directory.

.. code-block:: bash

    python setup.py install

Testing
^^^^^^^

The :code:`pytest` module is used for testing. In order to run and :code:`pytest`, you will need to ensure it is
installed on your machine. Consult the `pytest <https://docs.pytest.org/en/latest/>`_ website for more information. To
run the suite of tests for :code:`qustop`, run the following command in the root directory of this project:

.. code-block:: bash

    pytest --cov-report term-missing --cov=qustop tests/

Contributing
^^^^^^^^^^^^

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
`contributing guide <https://github.com/vprusso/qustop/blob/master/.github/CONTRIBUTING.md>`_.

Citing
^^^^^^

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
       title        = {qustop: A {P}ython package for investigating quantum state optimization, version 1.0.0},
       howpublished = {\url{https://github.com/vprusso/qustop}},
       month        = May,
       year         = 2021,
       doi          = {XXX}
     }
