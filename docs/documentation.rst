===================
Build Documentation
===================

exafmm-t depends on **doxygen**, **sphinx** and **breathe** to generate this documentation. 
To faciliate generating C++ API documentation, we use **exhale**, a Sphinx extension,
to automate launching Doxygen and calling Sphinx to create documentation based on Doxygen xml output.

To build this documentation locally, you need to install doxygen with your package manager, install other dependencies using
``pip install -r docs/requirements.txt``, and then use the following commands:

.. code-block:: bash

   $ cd docs
   $ make html

The HTML documentation will be generated in ``docs/_build/html`` directory.

We also have set up Travis CI to automatically deploy the documentation to Github Pages.