#ifndef laplace_cuda_h
#define laplace_cuda_h
#include "exafmm_t.h"

namespace exafmm_t {
   void P2PGPU(real_t* trg_coord, real_t* trg_val, real_t* src_coord, real_t* src_val, int leafs_cnt, int ncrit, int adj_cnt);
}
#endif
