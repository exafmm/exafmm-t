========
Examples
========

C++ Examples
------------

There are three major classes in exafmm-t:

- ``Body<T>``: The class for bodies (particles).
- ``Node<T>``: The class for nodes in the octree.
- ``Fmm``: The FMM class.

The choice of template parameter ``T`` depends on the data type of the potential:
``T`` should be set to ``real_t`` for real-valued kernels (ex. Laplace and modified Helmholtz),
and set to ``complex_t`` for complex-valued kernels (ex. Helmholtz).

exafmm-t uses double precision by default, i.e., ``real_t`` and ``complex_t`` are mapped to ``double`` and ``std::complex<double>`` respectively.
If you want to use single precision, you should still use ``real_t`` and ``complex_t`` in your code,
and add ``-DFLOAT`` to your compiler flags which predefines the macro ``FLOAT`` as true.

All exafmm-t's types, classes and functions are in ``exafmm_t`` namespace.
API documentation can be found in the last section.

Let's solve a Laplace N-body problem as an example, we first need to create ``sources`` and ``targets``.
Here we create 100,000 sources and targets that are randomly distributed in a cube from -1 to 1.
Their type ``Bodies`` is a STL vector of ``Body``.

.. code-block:: cpp
   
   using exafmm_t::real_t;
   std::random_device rd;
   std::mt19937 gen(rd());  // random number generator
   std::uniform_real_distribution<> dist(-1.0, 1.0);
   int ntargets = 100000;
   int nsources = 100000;
   
   exafmm_t::Bodies<real_t> sources(nsources);
   for (int i=0; i<nsources; i++) {
     sources[i].ibody = i;
     sources[i].q = dist(gen);        // charge
     for (int d=0; d<3; d++)
       sources[i].X[d] = dist(gen);   // location
   }

   exafmm_t::Bodies<real_t> targets(ntargets);
   for (int i=0; i<ntargets; i++) {
     targets[i].ibody = i;
     for (int d=0; d<3; d++)
       targets[i].X[d] = dist(gen);   // location
   }

Next, we need to create an FMM instance ``fmm`` for Laplace kernel, and set the order of expansion and ncrit.
We use the former to control the accuracy and the latter to balance the workload between near-field and far-field.

.. code-block:: cpp

   int P = 8;         // expansion order
   int ncrit = 400;   // max number of bodies per leaf
   exafmm_t::LaplaceFmm fmm(P, ncrit);

We can then build and balance the octree. The variable ``nodes`` represents the tree, whose type is ``Nodes``, a STL vector of ``Node``.
To facilitate creating lists and evaluation, we also store a vector of leaf nodes - ``leafs`` and a vector of non-leaf nodes - ``nonleafs``.
Their type ``NodePtrs`` is a STL vector of ``Node*``.

.. code-block:: cpp

   exafmm_t::get_bounds(sources, targets, fmm.x0, fmm.r0);
   exafmm_t::NodePtrs<real_t> leafs, nonleafs;
   exafmm_t::Nodes<real_t> nodes = exafmm_t::build_tree<real_t>(sources, targets, leafs, nonleafs, fmm);
   exafmm_t::balance_tree<real_t>(nodes, sources, targets, leafs, nonleafs, fmm);

Next, we can build lists and pre-compute invariant matrices.

.. code-block:: cpp

   exafmm_t::init_rel_coord();        // compute all possible relative positions of nodes for each FMM operator
   exafmm_t::set_colleagues(nodes);   // find colleague nodes
   exafmm_t::build_list(nodes, fmm);  // create list for each FMM operator
   fmm.M2L_setup(nonleafs);           // an extra setup for M2L operator

Finally, we can use FMM to evaluate potentials and gradients

.. code-block:: cpp

   fmm.upward_pass(nodes, leafs);
   fmm.downward_pass(nodes, leafs);
   
After the downward pass, the calculated potentials and gradients are stored in the leaf nodes of the tree.
You can compute the error in L2 norm by comparing with direct summation:

.. code-block:: cpp

   std::vector<real_t> error = fmm.verify(leafs);
   std::cout << "potential error: " << error[0] << "\n"
             << "gradient error:  " << error[1] << "\n";

Other examples can be found in ``tests`` folder.

Python Examples
---------------

For simplicity, the name of our Python package is just ``exafmm``.
It has a separate module for each kernel: ``exafmm.laplace``, ``exafmm.helmholtz`` and ``exafmm.modified_helmholtz``.

Compare with C++ interface, exafmm-t's Python interface only exposes high-level APIs.
Now, the steps for tree construction, list construction and pre-computation are merged into one function called ``setup()``.
Also, the evaluation now only requires to call one function ``evalute()``.
Below are Python examples on Jupyter notebooks.

- `Laplace <https://nbviewer.jupyter.org/github/exafmm/exafmm-t/blob/master/examples/laplace_example.ipynb>`__
- `Helmholtz <https://nbviewer.jupyter.org/github/exafmm/exafmm-t/blob/master/examples/helmholtz_example.ipynb>`__
- `Modified Helmholtz <https://nbviewer.jupyter.org/github/exafmm/exafmm-t/blob/master/examples/modified_helmholtz_example.ipynb>`__