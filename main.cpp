#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#if COMPLEX
#include "laplace_c.h"
#include "precompute_c.h"
#else
#include "laplace.h"
#include "precompute.h"
#endif
#include "traverse.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  Profile::Enable(true);

  Profile::Tic("Total", true);
  Bodies sources = initBodies(args.numBodies, args.distribution, 0);
  Bodies targets = initBodies(args.numBodies, args.distribution, 0);
  NodePtrs leafs, nonleafs;
  Nodes nodes = buildTree(sources, targets, leafs, nonleafs, args);

  // balanced tree
  std::unordered_map<uint64_t, size_t> key2id;
  Keys keys = breadthFirstTraversal(&nodes[0], key2id);
  Keys bkeys = balanceTree(keys, key2id, nodes);
  Keys leafkeys = findLeafKeys(bkeys);
  nodes.clear();
  leafs.clear();
  nonleafs.clear();
  nodes = buildTree(sources, targets, leafs, nonleafs, args, leafkeys);
  MAXLEVEL = keys.size() - 1;

  // fill in coords and values for compatibility
  // remove this later
  initRelCoord();    // initialize relative coords
  Profile::Tic("Precomputation", true);
  Precompute();
  Profile::Toc();
  setColleagues(nodes);
  buildList(nodes);
  M2LSetup(nonleafs);
  upwardPass(nodes, leafs);
  downwardPass(nodes, leafs);
  Profile::Toc();
  RealVec error = verify(leafs);
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() << std::endl;
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< MAXLEVEL << std::endl;
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
  Profile::print();
  return 0;
}
