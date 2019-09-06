Installation
============

Dependencies
------------
* C++ Compiler (g++ and icpc have been tested)
* BLAS
* LAPACK
* `FFTW3 <http://www.fftw.org/download.html>`_


How to compile
--------------
Exafmm-t uses **autotools** as the build-system. Go to the root directory of exafmm-t and configure the build:

.. code-block:: bash

   $ cd exafmm-t
   $ ./configure

By default, the configure script will enable mpi and openmp options (if available), use the highest SIMD instruction set 
available on the CPU and enable double precision. Use ``./configure --help`` to see all available options.

After configuration, you can compile exafmm-t with:

.. code-block:: bash

   $ make

at the root directory of the repo.