===================
Build Documentation
===================

exafmm-t depends on **doxygen**, **sphinx** and **breathe** to generate this documentation. 
To faciliate generating C++ API documentation, we use **exhale**, a Sphinx extension,
to automate launching Doxygen and calling Sphinx to create documentation based on Doxygen xml output.

Follow these steps to build this documentation:

.. code-block:: bash

   $ cd docs
   $ make html

The HTML documentation will be generated in ``docs/_build/html`` directory.