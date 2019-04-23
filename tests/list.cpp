#include <cassert>
#include "args.h"
#include "build_list.h"
#include "build_tree.h"
#include "dataset.h"
#include "geometry.h"
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
using std::cout;
using std::endl;

int main(int argc, char** argv) {
  NSURF = 1;    // only need to store monopole
  Args args(argc, argv);
  Bodies sources = init_bodies(args.numBodies, args.distribution, 0, true);
  Bodies targets = init_bodies(args.numBodies, args.distribution, 5, false);
  
  get_bounds(sources, targets, XMIN0, R0);
  NodePtrs leafs, nonleafs;
  Nodes nodes = build_tree(sources, targets, XMIN0, R0, leafs, nonleafs, args);
  balance_tree(nodes, sources, targets, XMIN0, R0, leafs, nonleafs, args);
  init_rel_coord();
  set_colleagues(nodes);
  build_list(nodes);

  Node* root = nodes.data();
  P2M(leafs);
  M2M(root);
  P2L(nodes);
  M2P(leafs);
  P2P(leafs);
  M2L(nonleafs);
  L2L(root);
  L2P(leafs);

  print("number of sources", args.numBodies);
  print("checking leaf's potential...");
#pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<leafs.size(); ++i) {
    Node* leaf = leafs[i];
    assert(args.numBodies == leaf->trg_value[0]);
  }
  print("assertion passed!");
  return 0;
}
