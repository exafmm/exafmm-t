#include <iostream>
#include "bempp.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  // generate random coordinates and charges
  size_t N = 100000;
  real_t * src_coord = new real_t [3*N];
  real_t * src_value = new real_t [N];
  real_t * trg_coord = new real_t [3*N];
  real_t * trg_value = new real_t [N];
  for(size_t i=0; i<N; ++i) {
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    }
    src_value[i] = drand48();
  }

  // initialize global variables
  int threads = 6;
  init_FMM(threads);

  // setup FMM
  setup_FMM(N, src_coord, src_value, N, trg_coord);

  // run FMM
  run_FMM();

  // delete arrays
  delete[] src_coord;
  delete[] src_value;
  delete[] trg_coord;
  delete[] trg_value;
  
  return 0;
}
