#include <cmath>    // sin, cos
#include <cstdlib>  // drand48
#include "bempp_wrapper_laplace.h"

int main(int argc, char **argv) {
  // generate coordinates of sphere distribution and random charges
  int N = 1000000;
  double * src_coord = new double [3*N];
  double * src_value = new double [N];
  double * trg_coord = new double [3*N];
  double * trg_value = new double [4*N];
  double theta, phi, r = 1;
  for(int i=0; i<N; ++i) {
    src_value[i] = drand48() - 0.5;
    theta = drand48() * M_PI;
    phi = drand48() * 2 * M_PI;
    src_coord[3*i+0] = r * std::sin(theta) * std::cos(phi);
    src_coord[3*i+1] = r * std::sin(theta) * std::sin(phi);
    src_coord[3*i+2] = r * std::cos(theta);
    theta = drand48() * M_PI;
    phi = drand48() * 2 * M_PI;
    trg_coord[3*i+0] = r * std::sin(theta) * std::cos(phi);
    trg_coord[3*i+1] = r * std::sin(theta) * std::sin(phi);
    trg_coord[3*i+2] = r * std::cos(theta);
    for(int d=0; d<4; ++d) {
      trg_value[4*i+d] = 0.;
    }
  }

  // initialize global variables
  int p = 10;
  int maxlevel = 5;
  int threads = 6;
  init_FMM(p, maxlevel, threads);

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
