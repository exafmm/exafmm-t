#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"
#include "cufft.h"

namespace exafmm_t {
  void cuda_init_drivers();

  void P2PGPU(std::vector<real_t> leafs_coord, std::vector<int> leafs_coord_idx, std::vector<real_t> leafs_pt_src, std::vector<int> leafs_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val, int leafs_size);

  std::vector<real_t> M2LGPU(std::vector<int> &M2Ltargets_idx, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, AlignedVec &up_equiv, int M2Lsources_idx_size);
}
#endif
