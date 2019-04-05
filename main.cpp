#if FULL_TREE
#include "build_full_tree.h"
#else
#include "build_tree.h"
#endif
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
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t WAVEK;
#endif
}

using namespace exafmm_t;
using namespace std;
int main(int argc, char **argv) {
#if HELMHOLTZ
  WAVEK = 20;
#endif
  Args args(argc, argv);
#if HAVE_OPENMP
  omp_set_num_threads(args.threads);
#endif
  size_t N = args.numBodies;
  P = args.P;
  NSURF = 6*(P-1)*(P-1) + 2;

  start("Total");
  Bodies sources = init_bodies(args.numBodies, args.distribution, 0, true);
  Bodies targets = init_bodies(args.numBodies, args.distribution, 5, false);

  start("Build Tree");
  get_bounds(sources, targets, XMIN0, R0);
  NodePtrs leafs, nonleafs;
#if FULL_TREE
  MAXLEVEL = args.maxlevel;   // explicitly define the max level when constructing a full tree
  Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs);
#else
  Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs, args);
  balance_tree(nodes, sources, targets, XMIN0, R0, leafs, nonleafs, args);
#endif
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
  return 0;
}
