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
  std::vector<int> leafs_idx;
  std::vector<int> childs_idx;
  std::vector<int> nodes_depth;
  std::vector<int> nodes_idx;
  Bodies bodies = cube(args.numBodies, 0);
  std::vector<real_t> nodes_coord;
  Nodes nodes = buildTree(bodies, leafs_idx, args);
  MAXLEVEL = 0;
  for(size_t i=0; i<leafs_idx.size(); i++) {
    MAXLEVEL = std::max(MAXLEVEL, nodes[leafs_idx[i]].depth);
  }
  std::vector<std::vector<int>> nodes_by_level_idx(MAXLEVEL);
  std::vector<std::vector<int>> parent_by_level_idx(MAXLEVEL);
  std::vector<std::vector<int>> octant_by_level_idx(MAXLEVEL);
  for(int i=1;i<nodes.size();i++){
    nodes_by_level_idx[nodes[i].depth-1].push_back(nodes[i].idx);
    parent_by_level_idx[nodes[i].depth-1].push_back(nodes[i].parent->idx);
    octant_by_level_idx[nodes[i].depth-1].push_back(nodes[i].octant);
  }
  // fill in pt_coord, pt_src, correct coord for compatibility
  // remove this later
  std::vector<real_t> bodies_coord;
  RealVec upward_equiv(nodes.size()*NSURF);  
  RealVec dnward_equiv(nodes.size()*NSURF);
  std::vector<real_t> nodes_pt_src;
  std::vector<int> nodes_pt_src_idx;
  int nodes_pt_src_idx_cnt = 0;
  for(int i=0; i<nodes.size(); i++) {
    nodes_depth.push_back(nodes[i].depth);
    nodes_idx.push_back(nodes[i].idx);
    for(int d=0; d<3; d++) {
      nodes_coord.push_back(nodes[i].coord[d]);
    }

    nodes_pt_src_idx.push_back(nodes_pt_src_idx_cnt);
    if(nodes[i].IsLeaf()) {
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numBodies; B++) {
        bodies_coord.push_back(B->X[0]);
        bodies_coord.push_back(B->X[1]);
        bodies_coord.push_back(B->X[2]);
        nodes_pt_src.push_back(B->q);
        nodes_pt_src_idx_cnt ++;
      }
    }
  }
  nodes_pt_src_idx.push_back(nodes_pt_src_idx_cnt);
  std::vector<real_t> nodes_trg(nodes_pt_src.size()*4, 0.);
  std::vector<int> M2Lsources_idx, M2Ltargets_idx;
  initRelCoord();    // initialize relative coords
  Profile::Tic("Precomputation", true);
  Precompute();
  Profile::Toc();
  setColleagues(nodes);
  Profile::Tic("buildList", true);
  buildList(nodes, M2Lsources_idx, M2Ltargets_idx);
  Profile::Toc();
  fmmStepsGPU(nodes, leafs_idx, bodies_coord, nodes_pt_src, nodes_pt_src_idx, args.ncrit, upward_equiv, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, nodes_coord, M2Lsources_idx, M2Ltargets_idx, dnward_equiv, nodes_trg, nodes_depth, nodes_idx);

  downwardPass(nodes, leafs_idx, M2Lsources_idx, M2Ltargets_idx, bodies_coord, nodes_pt_src, nodes_pt_src_idx, args.ncrit, upward_equiv, dnward_equiv, nodes_trg,  nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, nodes_coord, nodes_depth, nodes_idx);
  Profile::Toc();
  RealVec error = verify(nodes, leafs_idx, bodies_coord, nodes_pt_src, nodes_pt_src_idx, nodes_trg);
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs_idx.size() << std::endl;
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< nodes[leafs_idx.back()].depth << std::endl;
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
  Profile::print();
  return 0;
}
