#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#include "laplace_cuda.h"
#include "laplace.h"
#include "precompute.h"
#include "traverse.h"
using namespace exafmm_t;
RealVec plummer(int);
RealVec nonuniform(int);

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  Profile::Enable(true);
  cuda_init_drivers();
  RealVec src_coord, src_value;

  Profile::Tic("Total", true);
  Bodies bodies = cube(args.numBodies, 0);
  std::vector<int> leafs_idx;
  Nodes nodes = buildTree(bodies, leafs_idx, args);
  MAXLEVEL = 0;
  for(size_t i=0; i<leafs_idx.size(); i++) {
    MAXLEVEL = std::max(MAXLEVEL, nodes[leafs_idx[i]].depth);
  }
  initRelCoord();    // initialize relative coords
  Profile::Tic("Precomputation", true);
  Precompute();
  Profile::Toc();
  setColleagues(nodes);
  std::vector<int> nodes_pt_src_idx, nodes_depth(nodes.size()), nodes_idx(nodes.size()), M2Lsources_idx, M2Ltargets_idx;
  std::vector<real_t> nodes_coord(nodes.size()*3), bodies_coord, nodes_pt_src;
  std::vector<std::vector<int>> nodes_by_level_idx(MAXLEVEL), parent_by_level_idx(MAXLEVEL), octant_by_level_idx(MAXLEVEL);
  Profile::Tic("buildList", true);
  buildList(nodes, M2Lsources_idx, M2Ltargets_idx, leafs_idx, nodes_pt_src_idx, nodes_depth, nodes_idx, nodes_coord, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, bodies_coord, nodes_pt_src);
  std::vector<real_t> nodes_trg(nodes_pt_src.size()*4, 0.);
  Profile::Toc();
  fmmStepsGPU(nodes, leafs_idx, bodies_coord, nodes_pt_src, nodes_pt_src_idx, args.ncrit, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, nodes_coord, M2Lsources_idx, M2Ltargets_idx, nodes_trg, nodes_depth, nodes_idx);
  Profile::Toc();
  RealVec error = verify(nodes, leafs_idx, bodies_coord, nodes_pt_src, nodes_pt_src_idx, nodes_trg);
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs_idx.size() << std::endl;
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< nodes[leafs_idx.back()].depth << std::endl;
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
  Profile::print();
  return 0;
}
