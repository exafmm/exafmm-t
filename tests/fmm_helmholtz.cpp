#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "build_tree.h"
#endif
#include "build_list.h"
#include "config.h"
#include "dataset.h"
#include "precompute_helmholtz.h"
#include "helmholtz.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 X0;
  real_t R0;
  real_t WAVEK;
}

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
#if HAVE_OPENMP
  omp_set_num_threads(args.threads);
#endif
  WAVEK = args.k;
  size_t N = args.numBodies;
  P = args.P;
  NSURF = 6*(P-1)*(P-1) + 2;
  print_divider("Parameters");
  args.print();

  print_divider("Time");
  start("Total");
  Bodies<complex_t> sources = init_sources<complex_t>(args.numBodies, args.distribution, 0);
  Bodies<complex_t> targets = init_targets<complex_t>(args.numBodies, args.distribution, 5);

  start("Build Tree");
  get_bounds(sources, targets, X0, R0);
  NodePtrs<complex_t> leafs, nonleafs;
#if NON_ADAPTIVE
  MAXLEVEL = args.maxlevel;   // explicitly define the max level when constructing a full tree
  Nodes<complex_t> nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs);
#else
  Nodes<complex_t> nodes = build_tree(sources, targets, X0, R0, leafs, nonleafs, args);
  balance_tree(nodes, sources, targets, X0, R0, leafs, nonleafs, args);
#endif
  stop("Build Tree");

  init_rel_coord();

  start("Precomputation");
  helmholtz::precompute();
  stop("Precomputation");

  start("Build Lists");
  set_colleagues(nodes);
  build_list(nodes);
  stop("Build Lists");

  helmholtz::M2L_setup(nonleafs);
  helmholtz::upward_pass(nodes, leafs);
  helmholtz::downward_pass(nodes, leafs);
  stop("Total");

  RealVec error = helmholtz::verify(leafs);
  print_divider("Error");
  print("Potential Error", error[0]);
  print("Gradient Error", error[1]);

  print_divider("Tree");
  print("Root Center x", X0[0]);
  print("Root Center y", X0[1]);
  print("Root Center z", X0[2]);
  print("Root Radius R", R0);
  print("Tree Depth", MAXLEVEL);
  print("Leaf Nodes", leafs.size());
  print("Helmholtz kD", 2*R0*WAVEK);
  return 0;
}
