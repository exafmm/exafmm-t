#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "build_tree.h"
#endif
#include "build_list.h"
#include "dataset.h"
#include "helmholtz.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  print_divider("Parameters");
  args.print();

#if HAVE_OPENMP
  omp_set_num_threads(args.threads);
#endif

  print_divider("Time");
  Bodies<complex_t> sources = init_sources<complex_t>(args.numBodies, args.distribution, 0);
  Bodies<complex_t> targets = init_targets<complex_t>(args.numBodies, args.distribution, 5);

  start("Total");
  HelmholtzFmm fmm(args.P, args.ncrit, args.maxlevel, args.k);

  start("Build Tree");
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  NodePtrs<complex_t> leafs, nonleafs;
#if NON_ADAPTIVE
  Nodes<complex_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
#else
  Nodes<complex_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  balance_tree(nodes, sources, targets, leafs, nonleafs, fmm);
#endif
  stop("Build Tree");

  init_rel_coord();

  start("Build Lists");
  set_colleagues(nodes);
  build_list(nodes, fmm);
  stop("Build Lists");

  start("Precomputation");
  fmm.precompute();
  stop("Precomputation");

  start("M2L Setup");
  fmm.M2L_setup(nonleafs);
  stop("M2L Setup");
  fmm.upward_pass(nodes, leafs);
  fmm.downward_pass(nodes, leafs);
  stop("Total");

  bool sample = (args.numBodies >= 10000);
  RealVec err = fmm.verify(leafs, sample);
  print_divider("Error");
  print("Potential Error L2", err[0]);
  print("Gradient Error L2", err[1]);

  print_divider("Tree");
  print("Root Center x", fmm.x0[0]);
  print("Root Center y", fmm.x0[1]);
  print("Root Center z", fmm.x0[2]);
  print("Root Radius R", fmm.r0);
  print("Tree Depth", fmm.depth);
  print("Leaf Nodes", leafs.size());
  print("Helmholtz kD", 2*fmm.r0*fmm.wavek);

  return 0;
}
