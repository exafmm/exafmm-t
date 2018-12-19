#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"

namespace exafmm_t {
  void cuda_init_drivers();

  void P2PGPU(std::vector<int> leafs_idx, std::vector<real_t> nodes_coord, std::vector<int> nodes_coord_idx, std::vector<real_t> nodes_pt_src, std::vector<int> nodes_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val);

  void HadmardGPU(std::vector<int> M2Ltargets_idx, std::vector<real_t> nodes_up_equiv_fft, std::vector<int> M2Llist_start_idx, std::vector<int> M2Llists, std::vector<int> M2LRelPos_start_idx, std::vector<int> M2LRelPoss, std::vector<real_t> mat_M2L_Helper, std::vector<real_t> &check);
}
#endif
