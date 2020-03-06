#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  RealVec surface(int p, real_t r0, int level, real_t* c, real_t alpha, bool is_mapping=false);

  RealVec convolution_grid(int p, real_t r0, int level, real_t* c);

  int hash(ivec3& coord);

  void init_rel_coord(int max_r, int min_r, int step, Mat_Type t);

  void generate_M2L_index_map();

  void init_rel_coord();
} // end namespace
#endif
