### bempp wrapper

This wrapper creates a simple interface between bempp and exafmm-t.

- `bempp_wrapper.cpp`: wrapper's source file, wrapping exafmm-t's functions with interface only consisting C basis types and their pointer types.
- `bempp_wrapper_[KERNEL].h`: wrapper's header file, users need to include this header in order to use the functions in the wrapper.
- `bempp_example_[KERNEL].cpp`: examples of using the wrapper.

How to compile:
- `./configure` in the top directory of exafmm-t
- `cd wrappers/bempp` go to this folder
- `make libbempp_exafmm_[KERNEL].a` compile wrapper source file to create static libraries
- `make bempp_[KERNEL]` compile example codes and link to libraries

Comments:
- The tree used here is non-adaptive. (see `/include/build_non_adaptive_tree.h`)
- FMM parameters (p, maxlevel, threads) and bodies' distribution can be changed in `bempp_example_[KERNEL].cpp`.
