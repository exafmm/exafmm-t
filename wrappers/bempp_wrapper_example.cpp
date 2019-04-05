#include <iostream>
#include "bempp_wrapper.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  // generate random coordinates and charges
  size_t N = 100000;
  real_t * src_coord = new real_t [3*N];
  value_t * src_value = new value_t [N];
  real_t * trg_coord = new real_t [3*N];
  value_t * trg_value = new value_t [4*N];
  for(size_t i=0; i<N; ++i) {
#if COMPLEX
    src_value[i] = {drand48(), drand48()};
#else
    src_value[i] = drand48();
#endif
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    }
    for(int d=0; d<3; ++d) {
#if COMPLEX
      trg_value[4*i+d] = {0.,0.};
#else
      trg_value[4*i+d] = 0.;
#endif
    }
  }

  // initialize global variables
  int threads = 6;
  init_FMM(threads);

  // setup FMM
  setup_FMM(N, src_coord, N, trg_coord);

  // run FMM
  run_FMM(src_value, trg_value);

  // check accuracy
  verify_FMM(N, src_coord, src_value,
             N, trg_coord, trg_value);

  // delete arrays
  delete[] src_coord;
  delete[] src_value;
  delete[] trg_coord;
  delete[] trg_value;
  
  return 0;
}
