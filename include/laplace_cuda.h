#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"

namespace exafmm_t {
  void cuda_init_drivers();

  void P2PGPU(std::vector<int> leafs_idx, std::vector<real_t> nodes_coord, std::vector<int> nodes_coord_idx, std::vector<real_t> nodes_pt_src, std::vector<int> nodes_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val);
}
#endif
