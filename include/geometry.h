#ifndef geometry_h
#define geometry_h
#include "exafmm_t.h"

namespace exafmm_t {
  RealVec surface(int p, real_t* c, real_t alpha, int depth);

  RealVec u_check_surf(real_t* c, int depth);

  RealVec u_equiv_surf(real_t* c, int depth);

  RealVec d_check_surf(real_t* c, int depth);

  RealVec d_equiv_surf(real_t* c, int depth);

  RealVec conv_grid(real_t* c, int depth);
} // end namespace
#endif
