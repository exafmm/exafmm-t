#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  RealVec surface(int p, real_t* c, real_t alpha, int level, bool is_mapping=false);

  RealVec convolution_grid(real_t* c, int level);

  int hash(ivec3& coord);

  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t);

  void init_rel_coord();

  std::vector<std::vector<int>> map_matrix_index();
} // end namespace
#endif
