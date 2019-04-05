#include <iostream>
#include "bempp_wrapper_laplace.h"

int main(int argc, char **argv) {
  // generate random coordinates and charges
  size_t N = 100000;
  double * src_coord = new double [3*N];
  double * src_value = new double [N];
  double * trg_coord = new double [3*N];
  double * trg_value = new double [4*N];
  for(size_t i=0; i<N; ++i) {
    src_value[i] = drand48();
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    }
    for(int d=0; d<3; ++d) {
      trg_value[4*i+d] = 0.;
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
