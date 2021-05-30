# History

As a bit of history of this project, it started in 2008 with [`PyFMM`](https://github.com/barbagroup/pyfmm), a 2D serial prototype in Python; then followed `PetFMM` in 2009, a PETSc-based parallel code with heavy templating [@cruz2011petfmm]; the third effort was [`GemsFMM`](https://github.com/barbagroup/gemsfmm) in 2010, a serial code with CUDA kernels for execution on GPUs [@yokota2011gems].
Another student in the group felt that [`exafmm-alpha`](https://github.com/exafmm/exafmm-alpha) overused class inheritance and was inadequate for his application, so he and a collaborator re-used only the kernels and implemented another tree construction in [`fmmtl`](https://github.com/ccecka/fmmtl).
The first author of this paper began working with `exafmm-alpha` in 2017, but at this point the mission was clear: simplify and aim for reusability.
That work led to the sixth implementation of FMM in the group, still called [`exafmm`](https://github.com/exafmm/exafmm), but it is not what we present here and we haven't published about it. 
With this new version of ExaFMM (called `exafmm-t`), we aim to bring the FMM to a broader audience and to increased scientific impact.

### Versions

#### exafmm-t
History: 2017/12/11 - Now  
Branch: gpu, vanilla-m2l  
Kernel: LaplaceKI, HelmholtzKI, YukawaKI 
Periodic: no  
SIMD: vec.h  
Thread: OpenMP loops and OpenMP tasks
MPI: none  
GPU: P2P, M2L  
Build: autoconf 
Wrapper: none  
Plotting: none  

#### exafmm
History: 2017/03/03 - Now  
Branch: dev, learning  
Kernel: Laplace, LaplaceKI, Helmholtz, Stokes  
Periodic: yes  
SIMD: vec.h  
Thread: OpenMP tasks  
MPI: HOT (global histogram sort)  
GPU: no  
Build: autoconf  
Wrapper: none  
Plotting: Python  

#### exafmm-alpha > exafmm-beta
History: 2012/07/21 - 2017/03/01  
Branch: develop, sc16  
Kernel: Laplace, Helmholtz, BiotSavart, Van der Waals  
Periodic: yes  
SIMD: Agner's vectorclass  
Thread: OpenMP, Cilk, TBB, Mthreads  
MPI: ORB (bisection, octsection)  
GPU: separate code (Bonsai hack)  
Build: autoconf & CMake  
Wrapper: CHARMM, GROMACS, general MD, PetIGA  
Plotting: none  

#### exafmm-alpha/old + vortex_method > exafmm-alpha
History: 2010/12/22 - 2012/07/21  
Branch: none  
Kernel: Laplace, Van der Waals, Biot Savart, Stretching, Gaussian  
Periodic: yes  
SIMD: none  
Thread: QUARK  
MPI: ORB (global nth_element)  
GPU: offload all kernels  
Build: Makefile  
Wrapper: MR3 compatible MD  
Plotting: VTK  

#### old_fmm_bem
History: mid 2010 - late 2010  
Branch: none  
Kernel: Laplace, Laplace Gn, Helmholtz  
Periodic: no  
SIMD: none  
Thread: none  
MPI: Allgather LET  
GPU: offload all kernels  
Build: Makefile  
Wrapper: none  
Plotting: none  

#### old_fmm_vortex
History: early 2010 - mid 2010  
Branch: none  
Kernel: Laplace, Biot Savart, Stretching (transpose,mixed), Gaussian  
Periodic: yes (shear, channel)  
SIMD: none  
Thread: none  
MPI: Allgather LET  
GPU: offload all kernels  
Build: Makefile  
Wrapper: none  
Plotting: none  

#### Other FMM codes
[ASKIT](http://padas.ices.utexas.edu/libaskit/)  
[Bosnsai](https://github.com/treecode/Bonsai)  
[ChaNGa](https://github.com/N-BodyShop/changa/wiki/ChaNGa)  
[FDPS](https://github.com/FDPS/FDPS)  
[KIFMM](https://cs.nyu.edu/~harper/kifmm3d/documentation/index.html)  
[KIFMM new](https://github.com/jeewhanchoi/kifmm--hybrid--double-only)  
[Modylas](https://github.com/rioyokotalab/modylas)  
[PKIFMM](https://github.com/roynalnaruto/FMM_RPY_BROWNIAN/tree/master/pkifmm)  
[PEPC](http://www.fz-juelich.de/ias/jsc/EN/AboutUs/Organisation/ComputationalScience/Simlabs/slpp/SoftwarePEPC/_node.html)  
[pfalcON](https://pfalcon.lip6.fr)  
[PVFMM](https://github.com/dmalhotra/pvfmm)  
[Salmon treecode](https://github.com/rioyokotalab/salmon_treecode)  
[ScaFaCos](http://www.scafacos.de)  
[ScalFMM](http://people.bordeaux.inria.fr/coulaud/Softwares/scalFMM.html)
