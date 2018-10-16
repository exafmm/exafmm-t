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
  Bodies bodies = initBodies(args.numBodies, args.distribution, 0);
  std::vector<Node*> leafs, nonleafs;
  Nodes nodes = buildTree(bodies, leafs, nonleafs, args);

  // balanced tree
  std::unordered_map<uint64_t, size_t> key2id;
  Keys keys = breadthFirstTraversal(&nodes[0], key2id);
  Keys bkeys = balanceTree(keys, key2id, nodes);
  Keys leafkeys = findLeafKeys(bkeys);
  nodes.clear();
  leafs.clear();
  nonleafs.clear();
  nodes = buildTree(bodies, leafs, nonleafs, args, leafkeys);
  MAXLEVEL = keys.size() - 1;

  // fill in coords and values for compatibility
  // remove this later
  for(int i=0; i<nodes.size(); i++) {
    for(int d=0; d<3; d++) {
      nodes[i].coord[d] = nodes[i].X[d] - nodes[i].R;
    }
    if(nodes[i].IsLeaf()) {
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numSources; B++) {
        nodes[i].src_coord.push_back(B->X[0]);
        nodes[i].src_coord.push_back(B->X[1]);
        nodes[i].src_coord.push_back(B->X[2]);
        nodes[i].src_value.push_back(B->q);
      }
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numTargets; B++) {
        nodes[i].trg_coord.push_back(B->X[0]);
        nodes[i].trg_coord.push_back(B->X[1]);
        nodes[i].trg_coord.push_back(B->X[2]);
      }
    }
  }
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
