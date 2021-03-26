## Getting started

### Installing

1. Ensure you have python 3.9 or greater.

    See [Installing Python 3 on Linux](https://docs.python-guide.org/starting/install3/linux/) 
    for a guide on how to install python.

2. Consider using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).

3. The preferred way to install the `qustop` package is via `pip`.

```
pip install qustop
```

Alternatively, to install, you may also run the following command from the
top-level package directory.


```
python setup.py install
```

### Testing

The `pytest` module is used for testing. In order to run and `pytest`, you will need to ensure it is installed on your 
machine. Consult the [pytest](https://docs.pytest.org/en/latest/) website for more information. To run the suite of 
tests for `toqito`, run the following command in the root directory of this project:

```
pytest -v --cov=qustop --cov-report=term --cov-report=xml qustop
```
    
### Contributing

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

A detailed overview on how to contribute can be found in the
[contributing guide](https://github.com/vprusso/qustop/blob/master/.github/CONTRIBUTING.md).

### Citing

You can cite `qustop` using the following DOI: [10.5281/zenodo.3699578](https://zenodo.org/record
/3699578>).

If you are using the `qustop` software package in research work, please
include an explicit mention of `qustop` in your publication. Something
along the lines of:

    To solve problem "X" we used `qustop`; a package for studying various
    quantum state optimization scenarios

A BibTeX entry that you can use to cite :code:`toqito` is provided here:


    @misc{qustop,
       author       = {Vincent Russo},
       title        = {qustop: A {P}ython toolkit for quantum state optimization, version 0.0.1},
       howpublished = {\url{https://github.com/vprusso/qustop}},
       month        = Mar,
       year         = 2021,
       doi          = {10.5281/zenodo.3699578}
     }
