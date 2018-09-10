#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  RealVec surface(int p, real_t* c, real_t alpha, int depth);

  RealVec conv_grid(real_t* c, int depth);
} // end namespace
#endif
