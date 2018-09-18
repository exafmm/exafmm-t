#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"

namespace exafmm_t {
void potentialP2PGPU(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);
}
#endif
