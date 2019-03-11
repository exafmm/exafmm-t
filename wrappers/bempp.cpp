#include <iostream>
#include <omp.h>
#include "build_tree.h"
#include "build_list.h"
#include "config.h"
#include "dataset.h"
#if HELMHOLTZ
#include "helmholtz.h"
#else
#include "laplace.h"
#endif
#include "traverse.h"

namespace exafmm_t {
  Args args;
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif

  extern "C" void init_FMM(int threads) {
    P = 16; 
    NSURF = 6*(P-1)*(P-1) + 2;
#if HELMHOLTZ
    MU = 20;
#endif
    args.P = P;
    args.ncrit = 320;
    args.threads = threads;
#if HAVE_OPENMP
    omp_set_num_threads(args.threads);
#endif
  }

  extern "C" Bodies array_to_bodies(size_t count, real_t* coord, real_t* value, bool is_source=true) {
    Bodies bodies(count);
    for (size_t i=0; i<count; ++i) {
      for (int d=0; d<3; ++d) {
        bodies[i].X[d] = coord[3*i+d];
      }
      if (is_source)
        bodies[i].q = value[i];
    }
    return bodies;
  }

  // build 2:1 balanced tree, precompute invariant matrices, build interaction lists
  extern "C" Nodes setup_FMM(Bodies& sources, Bodies& targets, NodePtrs& leafs) {
    start("Build Tree");
    get_bounds(sources, targets, XMIN0, R0);
    NodePtrs nonleafs;
    Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs, args);
    balance_tree(nodes, sources, targets, XMIN0, R0, leafs, nonleafs, args);
    stop("Build Tree");

    init_rel_coord();
    start("Precomputation");
    precompute();
    stop("Precomputation");
    start("Build Lists");
    set_colleagues(nodes);
    build_list(nodes);
    stop("Build Lists");

    M2L_setup(nonleafs);
    return nodes;
  }

  extern "C" void run_FMM(Nodes& nodes, NodePtrs& leafs) {
    start("Total");
    upward_pass(nodes, leafs);
    downward_pass(nodes, leafs);
    stop("Total");
 
    RealVec error = verify(leafs);
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
    std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : " << leafs.size() << std::endl;
    std::cout << std::setw(20) << std::left << "Tree Depth" << " : " << MAXLEVEL << std::endl;
  }
}
