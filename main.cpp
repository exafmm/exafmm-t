#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#elif COMPLEX
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

  Profile::Tic("Total");
  Bodies sources = initBodies(args.numBodies, args.distribution, 0);
  Bodies targets = initBodies(args.numBodies, args.distribution, 0);
  Profile::Tic("Build Tree");
  getBounds(sources, targets, Xmin0, R0);
  NodePtrs leafs, nonleafs;
  Nodes nodes = buildTree(sources, targets, Xmin0, R0, leafs, nonleafs, args);
  balanceTree(nodes, sources, targets, Xmin0, R0, leafs, nonleafs, args);
  Profile::Toc();
  initRelCoord();
  Profile::Tic("Precomputation");
  Precompute();
  Profile::Toc();
  Profile::Tic("Build Lists");
  setColleagues(nodes);
  buildList(nodes);
  Profile::Toc();
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
