#include "exafmm_t.h"
using namespace exafmm_t;
extern "C" void init_FMM(int threads);
extern "C" void setup_FMM(int src_count, real_t* src_coord,
                          int trg_count, real_t* trg_coord);
extern "C" void run_FMM(real_t* src_value, real_t* trg_value);
extern "C" void verify_FMM(int src_count, real_t* src_coord, real_t* src_value,
                           int trg_count, real_t* trg_coord, real_t* trg_value);
