#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"
#include "geometry.h"
#include <cuda_runtime.h>
#include "cufft.h"
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace exafmm_t {
  void cuda_init_drivers();

  void fmmStepsGPU(Nodes& nodes, std::vector<int> &leafs_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, int ncrit, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx, std::vector<real_t> &nodes_coord, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_depth, std::vector<int> &nodes_idx);
}

#endif

