#include <cassert>
#include "args.h"
#include "build_tree.h"
#include "dataset.h"
#include "print.h"
#include "test.h"

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

int main(int argc, char** argv) {
  NSURF = 1;    // only need to store monopole
  Args args(argc, argv);
  Bodies sources = init_bodies(args.numBodies, args.distribution, 0, true);
  Bodies targets = init_bodies(args.numBodies, args.distribution, 5, false);
  
  get_bounds(sources, targets, XMIN0, R0);
  NodePtrs leafs, nonleafs;
  Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs, args);
  balance_tree(nodes, sources, targets, XMIN0, R0, leafs, nonleafs, args);

  Node* root = nodes.data();
  P2M(leafs);
  M2M(root);

  print("number of sources", args.numBodies);
  print("root's monopole", root->up_equiv[0]);
  assert(args.numBodies == root->up_equiv[0]);
  return 0;
}
