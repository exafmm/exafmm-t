#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#include "laplace_cuda.h"
#include "laplace.h"
#include "precompute.h"
#include "traverse.h"
#include "profile.h"

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
  std::vector<Node*> leafs, nonleafs;
  Nodes nodes = buildTree(bodies, leafs, nonleafs, args);
  MAXLEVEL = 0;
  for(size_t i=0; i<leafs.size(); i++) {
    MAXLEVEL = std::max(MAXLEVEL, leafs[i]->depth);
  }
  // fill in pt_coord, pt_src, correct coord for compatibility
  // remove this later
  int leafs_cnt = 0;
  std::vector<int> leafs_idx;
  std::vector<int> nonleafs_idx;
  std::vector<real_t> nodes_coord;
  RealVec upward_equiv(nodes.size()*NSURF);  
  RealVec dnward_equiv(nodes.size()*NSURF);
  std::vector<real_t> nodes_pt_src;
  std::vector<int> nodes_pt_src_idx;
  int nodes_pt_src_idx_cnt = 0;
  for(int i=0; i<nodes.size(); i++) {
    for(int d=0; d<3; d++) {
      nodes[i].coord[d] = nodes[i].X[d] - nodes[i].R;
    }
    nodes_pt_src_idx.push_back(nodes_pt_src_idx_cnt);
    if(nodes[i].IsLeaf()) {
      leafs_cnt ++;
      leafs_idx.push_back(nodes[i].idx);
      for(Body* B=nodes[i].body; B<nodes[i].body+nodes[i].numBodies; B++) {
        nodes[i].pt_coord.push_back(B->X[0]);
        nodes[i].pt_coord.push_back(B->X[1]);
        nodes[i].pt_coord.push_back(B->X[2]);
	nodes[i].pt_src.push_back(B->q);

        nodes_coord.push_back(B->X[0]);
        nodes_coord.push_back(B->X[1]);
        nodes_coord.push_back(B->X[2]);
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
  buildList(nodes, M2Lsources_idx, M2Ltargets_idx);
  upwardPass(nodes, leafs_idx, nodes_coord, nodes_pt_src, nodes_pt_src_idx, args.ncrit, upward_equiv, nonleafs_idx);
  downwardPass(nodes, leafs, leafs_idx, M2Lsources_idx, M2Ltargets_idx, nodes_coord, nodes_pt_src, nodes_pt_src_idx, args.ncrit, upward_equiv, dnward_equiv, nodes_trg);
  Profile::Toc();
  RealVec error = verify(nodes, leafs_idx, nodes_coord, nodes_pt_src_idx, nodes_trg);
  std::cout << std::setw(20) << std::left << "Leaf Nodes" << " : "<< leafs.size() << std::endl;
  std::cout << std::setw(20) << std::left << "Tree Depth" << " : "<< leafs.back()->depth << std::endl;
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << error[0] << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << error[1] << std::endl;
  Profile::print();
  return 0;
}
