#include "build_tree.h"
#include "dataset.h"
#include "exafmm_t.h"
#include "local_essential_tree.h"
#include "partition.h"
#include "test.h"
#include "timer.h"

using namespace exafmm_t;

int main(int argc, char** argv) {
  Args args(argc, argv);
  startMPI(argc, argv);

  int n = args.numBodies;
  Bodies<real_t> sources = init_sources<real_t>(n, args.distribution, MPIRANK);
  Bodies<real_t> targets = init_targets<real_t>(n, args.distribution, MPIRANK+10);

  vec3 x0;
  real_t r0;
  allreduceBounds(sources, targets, x0, r0);
  std::vector<int> offset;   // based on the distribution of sources

  // partition information
  printMPI("before partitioning");
  int nsrcs = sources.size();
  int ntrgs = targets.size();
  printMPI("num of sources", nsrcs);
  printMPI("num of targets", ntrgs);
  partition(sources, targets, x0, r0, offset, args.maxlevel);
  printMPI("after partitioning");
  nsrcs = sources.size();
  ntrgs = targets.size();
  printMPI("num of sources", nsrcs);
  printMPI("num of targets", ntrgs);

  if (MPIRANK == 0) {
    std::cout << "Hilbert key offset for each rank : ";
    for (auto i : offset) std::cout << i << " ";
    std::cout << std::endl;
  }

  // build tree
  DummyFmm<real_t> fmm(args.ncrit);
  fmm.depth = args.maxlevel;
  fmm.x0 = x0;
  fmm.r0 = r0;
  NodePtrs<real_t> leafs, nonleafs;  
  printMPI("build tree locally");
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  printMPI("num of nodes", nodes.size());

  // upward pass
  Node<real_t>* root = nodes.data();
  fmm.P2M(leafs);
  fmm.M2M(root);
  assert(nsrcs == root->up_equiv[0]);
  printMPI("monopole check passed!");



  // build LET
  localEssentialTree(sources, targets, nodes, leafs, nonleafs,
                     fmm, offset);

  stopMPI();
}
