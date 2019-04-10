#ifndef traverse_h
#define traverse_h
#include "exafmm_t.h"
#include "profile.h"

namespace exafmm_t {
  void upwardPass(Nodes& nodes, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, int ncrit, RealVec &upward_equiv, std::vector<int> &nonleafs_idx) {
    Profile::Tic("P2M", false, 5);
    P2M(nodes, leafs_idx, nodes_coord, nodes_pt_src, nodes_pt_src_idx, ncrit, upward_equiv);
    Profile::Toc();
    Profile::Tic("M2M", false, 5);
    M2M(nodes, upward_equiv, nonleafs_idx);
    Profile::Toc();
  }

  void downwardPass(Nodes& nodes, std::vector<int> &leafs_idx, std::vector<int> &nonleafs_idx, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, int ncrit, RealVec &upward_equiv, RealVec &dnward_equiv, std::vector<real_t> &nodes_trg, std::vector<int> &childs_idx) {
    Profile::Tic("P2L", false, 5);
    P2L(nodes, dnward_equiv, nodes_pt_src, nodes_pt_src_idx, nodes_coord);
    Profile::Toc();
    Profile::Tic("M2P", false, 5);
    M2P(nodes, leafs_idx, upward_equiv, nodes_trg, nodes_pt_src_idx, nodes_coord);
    Profile::Toc();
    Profile::Tic("P2P", false, 5);
    P2P(nodes, leafs_idx, nodes_coord, nodes_pt_src, nodes_trg, nodes_pt_src_idx, ncrit);
    Profile::Toc();
    Profile::Tic("M2L", false, 5);
    M2L(nodes, M2Lsources_idx, M2Ltargets_idx, upward_equiv, dnward_equiv);
    Profile::Toc();
    Profile::Tic("L2L", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    //L2L(nodes, &nonleafs_idx[0], dnward_equiv, childs_idx, 0);
    L2L(&nodes[0], dnward_equiv);
    Profile::Toc();
    Profile::Tic("L2P", false, 5);
    L2P(nodes, dnward_equiv, leafs_idx, nodes_trg, nodes_pt_src_idx, nodes_coord);
    Profile::Toc();
  }

  RealVec verify(Nodes &nodes, std::vector<int>& leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &nodes_trg) {
    int numTargets = 10;
    int stride = leafs_idx.size() / numTargets;
    int direct_target_pt_trg_size = 0;
    std::vector<int> direct_target_pt_trg_idx;
    for(size_t i=0; i<numTargets; i++) {
      direct_target_pt_trg_idx.push_back(direct_target_pt_trg_size);
      direct_target_pt_trg_size += 4*(nodes_pt_src_idx[leafs_idx[i*stride]+1]-nodes_pt_src_idx[leafs_idx[i*stride]]);
    }
    direct_target_pt_trg_idx.push_back(direct_target_pt_trg_size);
    RealVec direct_target_pt_trg(direct_target_pt_trg_size, 0.);
#pragma omp parallel for
    for(size_t i=0; i<numTargets; i++) {
      for(size_t j=0; j<leafs_idx.size(); j++) {
        int leaf_coord_size = 3*(nodes_pt_src_idx[leafs_idx[j]+1] - nodes_pt_src_idx[leafs_idx[j]]);
        int pt_coord_size = 3*(nodes_pt_src_idx[leafs_idx[i*stride]+1]-nodes_pt_src_idx[leafs_idx[i*stride]]);
        gradientP2P(&nodes_coord[nodes_pt_src_idx[leafs_idx[j]]*3], leaf_coord_size, &nodes_pt_src[nodes_pt_src_idx[leafs_idx[j]]], &nodes_coord[nodes_pt_src_idx[leafs_idx[i*stride]]*3], pt_coord_size, &direct_target_pt_trg[direct_target_pt_trg_idx[i]]);
      }
    }
    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
    for(size_t i=0; i<numTargets; i++) {
      p_norm += direct_target_pt_trg[direct_target_pt_trg_idx[i]]*direct_target_pt_trg[direct_target_pt_trg_idx[i]];
      p_diff += (direct_target_pt_trg[direct_target_pt_trg_idx[i]] - nodes_trg[nodes_pt_src_idx[leafs_idx[i*stride]]*4]) * (direct_target_pt_trg[direct_target_pt_trg_idx[i]] - nodes_trg[nodes_pt_src_idx[leafs_idx[i*stride]]*4]);
      for(int d=1; d<4; d++) {
        g_diff += (direct_target_pt_trg[direct_target_pt_trg_idx[i]+d] - nodes_trg[nodes_pt_src_idx[leafs_idx[i*stride]]*4+d]) * (direct_target_pt_trg[direct_target_pt_trg_idx[i]+d] - nodes_trg[nodes_pt_src_idx[leafs_idx[i*stride]]*4+d]);
        g_norm += direct_target_pt_trg[direct_target_pt_trg_idx[i]+d] * direct_target_pt_trg[direct_target_pt_trg_idx[i]+d];
      }
    } 
    RealVec l2_error(2);
    l2_error[0] = sqrt(p_diff/p_norm);
    l2_error[1] = sqrt(g_diff/g_norm);
    return l2_error;
  }
}//end namespace
#endif

