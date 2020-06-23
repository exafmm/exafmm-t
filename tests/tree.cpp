#include <cassert>
#include <unordered_set>
#include "args.h"
#include "build_tree.h"
#include "dataset.h"
#include "timer.h"
#include "test.h"

using namespace exafmm_t;

int main(int argc, char** argv) {
  Args args(argc, argv);
  Bodies<real_t> sources = init_sources<real_t>(args.numBodies, args.distribution, 0);
  Bodies<real_t> targets = init_targets<real_t>(args.numBodies, args.distribution, 5);

  // check isrcs and itrgs are correct in each leaf
  std::unordered_set<int> isrcs_set, itrgs_set;
  for (int i=0; i<args.numBodies; i++) {
    isrcs_set.insert(i);
    itrgs_set.insert(i);
  }

  DummyFmm<real_t> fmm(args.ncrit);  // p and nsurf are set to 1 in the constructor
  NodePtrs<real_t> leafs, nonleafs;
  get_bounds(sources, targets, fmm.x0, fmm.r0);
  Nodes<real_t> nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
  std::cout << "nodes size before tree balancing: " << nodes.size() << std::endl;
  balance_tree(nodes, sources, targets, leafs, nonleafs, fmm);
  std::cout << "nodes size after tree balancing: " << nodes.size() << std::endl;

  // verify ibody in leafs
  for (auto & leaf : leafs) {
    for (auto & isrc : leaf->isrcs) {
      int flag = isrcs_set.erase(isrc);
      assert(flag == 1);  // each isrc should be only erased once
    }
    for (auto & itrg : leaf->itrgs) {
      int flag = itrgs_set.erase(itrg);
      assert(flag == 1);
    }
  }
  assert(isrcs_set.empty());  // these sets should be empty
  assert(itrgs_set.empty());

  // upward pass and check monopole
  Node<real_t> * root = nodes.data();
  fmm.P2M(leafs);
  fmm.M2M(root);
  print("number of sources", args.numBodies);
  print("root's monopole", root->up_equiv[0]);
  assert(args.numBodies == root->up_equiv[0]);
  print("assertion passed!");
  return 0;
}
