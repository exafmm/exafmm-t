Build Documentation
===================

exaFMM use **doxygen**, **sphinx** and **breathe** to generate this documentation. Follow these steps to
build the documentation:

.. code-block:: bash

   $ cd docs
   $ doxygen Doxyfile
   $ make html

The HTML documentation will be generated in ``docs/build/html`` directory.