#ifndef bempp_wrapper_h
#define bempp_wrapper_h
#include "exafmm_t.h"
using namespace exafmm_t;

#if COMPLEX
  typedef complex_t value_t;
#else
  typedef real_t value_t;
#endif

extern "C" void init_FMM(int threads);

extern "C" void setup_FMM(int src_count, real_t* src_coord,
                          int trg_count, real_t* trg_coord);

extern "C" void run_FMM(value_t* src_value, value_t* trg_value);

extern "C" void verify_FMM(int src_count, real_t* src_coord, value_t* src_value,
                           int trg_count, real_t* trg_coord, value_t* trg_value);
#endif
