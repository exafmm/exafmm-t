============
Installation
============

Dependencies
------------
* a C++ compiler that supports OpenMP and C++11 standard (or newer).
* GNU Make
* BLAS
* LAPACK
* `FFTW3 <http://www.fftw.org/download.html>`_
* GFortran

Notes:

* GNU and Intel compilers have been tested. Compilers that use LLVM, such as clang, are not supported yet.
* We recommend to install OpenBLAS. The standard build also includes a full LAPACK library.
* There is no Fortran code in exafmm-t, but gfortran is required in the autotools macros that help configure BLAS and LAPACK libraries.

You can use the following commands to install these dependencies on Ubuntu:

.. code-block:: bash

   $ apt-get update
   $ apt-get -y install libopenblas-dev libfftw3-dev gfortran

Modify these commands accordingly if you are running other Linux distributions.


Install exafmm-t
----------------
This section is only necessary for the users who want to use exafmm-t in C++ applications.
Python users can skip to next section: :ref:`Install exafmm-t's Python package`.

exafmm-t uses **autotools** as the build-system. Go to the root directory of exafmm-t and configure the build:

.. code-block:: bash

   $ cd exafmm-t
   $ ./configure

By default, the configure script will use the most advanced SIMD instruction set 
available on the CPU and enable double precision option. Use ``./configure --help`` to see all available options.

After configuration, you can compile and run exafmm-t's tests with:

.. code-block:: bash

   $ make check

at the root directory of the repo.

Optionally, you can install the headers to the configured location:

.. code-block:: bash

   $ make install


Install exafmm-t's Python package
---------------------------------
exafmm-t relies on `pybind11 <https://github.com/pybind/pybind11>`_ to generate Python bindings.
It requires Python 2.7 or 3.x. To install the Python package, you need first to install OpenBLAS (as the choice of BLAS library),
in addition to the aforementioned dependencies.

Then install exafmm-t to your Python environment using **pip**:

.. code-block:: bash

   $ pip install git+https://github.com/exafmm/exafmm-t.git