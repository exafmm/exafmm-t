# exafmm-t 

[![Build Status](https://travis-ci.com/exafmm/exafmm-t.svg?branch=master)](https://travis-ci.com/exafmm/exafmm-t)

**exafmm-t** is a kernel-independent fast multipole method library for solving N-body problems.
It provides both C++ and Python APIs.
We use [pybind11](https://github.com/pybind/pybind11) to create Python bindings from C++ code.
exafmm-t aims to deliver compelling performance with a simple code design and a user-friendly interface.
It currently supports both potential and force calculation of Laplace, low-frequency Helmholtz and modified Helmholtz (Yukawa) kernel in 3D.
In addition, users can easily add other non-oscillatory kernels under exafmm-t's framework.

## Documentation

The full documentation is available [here](https://exafmm.github.io/exafmm-t).

Please use [GitHub issues](https://github.com/exafmm/exafmm-t/issues) for tracking bugs and requests.
To contribute to exafmm-t, please review [CONTRIBUTING](https://github.com/exafmm/exafmm-t/blob/master/CONTRIBUTING.md).