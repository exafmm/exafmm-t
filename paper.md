---
title: 'exafmm-t'
tags:
  - C++
  - Python
  - fast multipole method
  - low-rank approximation
  - high performance computing
authors:
 - name: Tingyu Wang
   orcid: 0000-0003-2520-0511
   affiliation: 1
 - name: Rio Yokota
   orcid: 0000-0001-7573-7873
   affiliation: 2
 - name: Lorena A. Barba
   orcid: 0000-0001-5812-2711
   affiliation: 1
affiliations:
 - name: The George Washington University
   index: 1
 - name: Tokyo Institute of Technology
   index: 2
date: 21 July 2020
bibliography: paper.bib
---

# Summary

The fast multipole method (FMM), recognized as one of the top 10 algorithms (@BoardSchulten2000) from the 20-th century in scientific computing,
is an algorithm that reduces the complexity of N-body problems from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ by approximating far-range interactions in a hierarchical way.
Originally developed for fast evaluation of the gravitational potential field, the FMM and its variants now have found many applications in different fields.

Over the past few decades, a plethora of highly optimized fast N-body implementations have emerged.
`Bonsai` (@bedorfSparseOctreeGravitational2012) is a gravitational treecode that runs entirely on GPU.
`ChaNGa` (@jetleyMassivelyParallelCosmological2008) is also a treecode that uses `Charm++` to automate dynamic load balancing.
In terms of FMM codes, `ScalFMM` (@blanchardScalFMMGenericParallel2015) implements the black-box FMM, a kernel-independent variant based on interpolation.
It comes with an option to use `StarPU` runtime system to handle heterogeneous task scheduling.
`PVFMM` (@malhotraPVFMMParallelKernel2015) can compute both particle and volume potentials using kernel-independent FMM (KIFMM).
`exafmm` (@yokotaFMMBasedDual2013) focuses on low accuracy optimizations via dual tree traversal.
However, one remaining challenge in the FMM community is to develop a well-established open-source software package, like FFTW for Fast Fourier transform,
that delivers compelling performance with a standard and easy-to-use interface.
This motivates us to develop `exafmm-t` to bring the FMM to a broader audience and more scientific applications.

`exafmm-t` is a parallel FMM library for solving N-body problems in 3D.
It implements the kernel-independent fast multipole method (@yingKernelindependentAdaptiveFast2004) and runs on multicore architectures. 
Currently, it supports both potential and force calculation of Laplace, low-frequency Helmholtz and modified Helmholtz (Yukawa) kernel; furthermore, users can add other non-oscillatory kernels with only modest effort in `exafmm-t`'s framework.
It is a header-only library written in C/C++ and also provides Python APIs using pybind11 (@pybind11).

`exafmm-t` is designed to be standard and lean.
First, it only uses C++ STL containers and depends on mature math libraries: BLAS, LAPACK and FFTW3.
Second, `exafmm-t` is moderately object-oriented, namely, the usage of encapsulation, inheritance and polymorphism is conservative or even minimal in the code.
As a result, the core library consists of around 6,000 lines of code, which is an order of magnitude shorter than many other FMM packages.

`exafmm-t` is concise but highly optimized.
To achieve competitive performance, our work combines techniques and optimizations from several past efforts.
On top of multi-threading using OpenMP, we further speed up the P2P operator (near-range interactions) using SIMD vectorization with SSE/AVX/AVX-512 compatibility;
we apply the cache optimization proposed in PVFMM (@malhotraPVFMMParallelKernel2015) to improve the performance of M2L operator (far-range interactions).
In addition, `exafmm-t` also allows users to pre-compute and store translation operators, which benefits applications that requires FMM evaluations iteratively.
The single-node performance of `exafmm-t` is on par with the state-of-the-art packages that we mentioned above.
We ran a benchmark that solves a Laplace N-body problem with 1 million randomly distributed particles on a workstation with a 14-core Intel i9-7940X CPU.
It took 0.95 and 1.48 seconds to obtain 7 and 10 digits of accuracy on potential respectively.

`exafmm-t` is also easy to extend.
Adding a new kernel only requires users to create a derived `FMM` class and provide the kernel function.
Last but not least, it offers high-level Python APIs to support Python applications.
Thanks to pybind11, most STL containers can be automatically converted to Python native data structures.
Since Python uses duck typing, we have to expose overloaded functions to different Python objects.
To avoid naming collision and keep a clean interface, we choose to create a Python module for each kernel under `exafmm-t`'s Python package, instead of adding suffixes to function and class names to identify types.

To demonstrate its capability, we are currently integrating `exafmm-t` into `Bempp-cl`, an open-source boundary element method (BEM) package in Python,
whose predecessor, `BEM++` (@smigajSolvingBoundaryIntegral2015), has enabled many acoustic and electromagnetic applications.
In BEM applications, computations are dominated by the dense matrix-vector multiplication (mat-vec) in each iteration.
`exafmm-t` reduces both time and memory cost of mat-vec to a linear complexity, thus makes `Bempp-cl` feasible to solve large-scale problems.


# References
