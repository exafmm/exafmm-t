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
  // global variables
  Args args;
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif
  Nodes NODES;
  NodePtrs LEAFS;

  Bodies array_to_bodies(size_t count, real_t* coord, real_t* value, bool is_source=true) {
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

  // Initialize args and set global constants
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

  // build 2:1 balanced tree, precompute invariant matrices, build interaction lists
  extern "C" void setup_FMM(int src_count, real_t* src_coord, real_t* src_value, 
                            int trg_count, real_t* trg_coord) {
    Bodies sources = array_to_bodies(src_count, src_coord, src_value);
    Bodies targets = array_to_bodies(trg_count, trg_coord, nullptr, false);

    start("Build Tree");
    get_bounds(sources, targets, XMIN0, R0);
    NodePtrs nonleafs;
    NODES = build_tree(sources, targets, XMIN0, R0, LEAFS, nonleafs, args);
    balance_tree(NODES, sources, targets, XMIN0, R0, LEAFS, nonleafs, args);
    stop("Build Tree");

    init_rel_coord();
    start("Precomputation");
    precompute();
    stop("Precomputation");
    start("Build Lists");
    set_colleagues(NODES);
    build_list(NODES);
    stop("Build Lists");

    M2L_setup(nonleafs);
  }

  extern "C" void run_FMM() {
    start("Total");
    upward_pass(NODES, LEAFS);
    downward_pass(NODES, LEAFS);
    stop("Total");
 
    RealVec error = verify(LEAFS);
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
    std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : " << LEAFS.size() << std::endl;
    std::cout << std::setw(20) << std::left << "Tree Depth" << " : " << MAXLEVEL << std::endl;
  }
}
