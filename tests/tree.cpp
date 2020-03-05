#include <cassert>
#include "args.h"
#include "build_tree.h"
#include "dataset.h"
#include "exafmm_t.h"
#include "timer.h"
#include "test.h"

using namespace exafmm_t;

int main(int argc, char** argv) {
  Args args(argc, argv);
  Bodies<real_t> sources = init_sources<real_t>(args.numBodies, args.distribution, 0);
  Bodies<real_t> targets = init_targets<real_t>(args.numBodies, args.distribution, 5);
  
  DummyFmm<real_t> fmm(args.ncrit);  // p and nsurf are set to 1 in the constructor

  NodePtrs<real_t> leafs, nonleafs;
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  balance_tree(nodes, sources, targets, leafs, nonleafs, fmm);

  Node<real_t>* root = nodes.data();
  fmm.P2M(leafs);
  fmm.M2M(root);

  print("number of sources", args.numBodies);
  print("root's monopole", root->up_equiv[0]);
  assert(args.numBodies == root->up_equiv[0]);
  return 0;
}
