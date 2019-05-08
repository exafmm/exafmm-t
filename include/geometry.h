#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  void surface(int p, real_t* c, real_t alpha, int depth, int offset, std::vector<real_t> &coord);
  RealVec conv_grid(real_t* c, int depth);
} // end namespace
#endif
