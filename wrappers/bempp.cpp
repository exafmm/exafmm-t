#include <iostream>
#include "build_tree.h"
#include "build_list.h"
#include "dataset.h"
#if HELMHOLTZ
#include "helmholtz.h"
#else
#include "laplace.h"
#endif
#include "traverse.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif

  extern "C" void run_FMM() {
#if HELMHOLTZ
    MU = 20;
#endif
    P = 6;
    NSURF = 6*(P-1)*(P-1) + 2;

    Args args;
    args.ncrit = 64;
    args.numBodies = 100000;
    args.P = P;
    args.threads = 4;
    args.distribution = "c";

    start("Total");
    Bodies sources = init_bodies(args.numBodies, args.distribution, 0);
    Bodies targets = init_bodies(args.numBodies, args.distribution, 0);

    start("Build Tree");
    get_bounds(sources, targets, XMIN0, R0);
    NodePtrs leafs, nonleafs;
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
    upward_pass(nodes, leafs);
    downward_pass(nodes, leafs);
    stop("Total");
 
    RealVec error = verify(leafs);
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
    std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() << std::endl;
    std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< MAXLEVEL << std::endl;
  }
}
