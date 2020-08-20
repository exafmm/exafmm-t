========
exafmm-t
========

exafmm-t is an open-source fast multipole method (FMM) library to simulate N-body interactions.
It implements the kernel-independent FMM and provides both C++ and Python APIs.
We use `pybind11 <https://github.com/pybind/pybind11>`__ to create Python bindings from the C++ source code.

Exafmm-t currently is a shared-memory implementation using OpenMP.
It aims to deliver competitive performance with a simple code design.
It also has the following features:

- offer high-level APIs in Python
- only use C++ STL containers
- support both single- and double-precision
- vectorization on near-range interactions
- cache optimization on far-range interactions