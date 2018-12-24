#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  RealVec surface(int p, real_t* c, real_t alpha, int level, bool is_mapping=false);

  RealVec convolution_grid(real_t* c, int level);
} // end namespace
#endif
