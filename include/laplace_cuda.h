#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"
#include <cuda_runtime.h>
#include "cufft.h"
#include "cublas_v2.h"

namespace exafmm_t {
  void cuda_init_drivers();
  
  void P2MGPU(std::vector<int> &leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &checkCoord, int trg_cnt, std::vector<real_t> &upward_equiv, std::vector<real_t> &r, std::vector<real_t> &leaf_xyz, int ncrit);

  void M2MGPU(RealVec &upward_equiv, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx);

  void P2PGPU(std::vector<int> &leafs_idx, std::vector<real_t> nodes_coord, std::vector<real_t> nodes_pt_src, std::vector<int> nodes_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val, int leafs_size, int ncrit);

  std::vector<real_t> M2LGPU(std::vector<int> &M2Ltargets_idx, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, AlignedVec &up_equiv, int M2Lsources_idx_size);

  void L2PGPU(RealVec &equivCoord, RealVec &dnward_equiv, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_trg, std::vector<int> &leafs_idx, std::vector<int> &nodes_pt_src_idx, int THREADS);
  void test_cuda(Nodes &nodes);
}

#endif
