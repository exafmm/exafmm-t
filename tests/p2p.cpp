#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#else
#include "laplace.h"
#include "precompute_laplace.h"
#endif
#include "traverse.h"
#include "exafmm_t.h"
// #include <math.h>

using namespace exafmm_t;
using namespace std;

void laplace_kernel(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  int trg_cnt = trg_coord.size()/3;
#pragma omp parallel for
  for(int t = 0; t < trg_cnt; t++) {
    real_t p = 0;
    for(int s = 0; s < src_value.size(); s++) {
      real_t r = 0;
      for(int k = 0; k < 3; k++) {
        r += (trg_coord[t*3+k] - src_coord[s*3+k])*(trg_coord[t*3+k] - src_coord[s*3+k]);
      }
      r = sqrt(r);
      if(r != 0) {
        p += src_value[s]/r;
      }
    }
    trg_value[t] = p / (4 * M_PI);
  }
}

void helmholtz_kernel(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
  // complex_t WAVEK = 0.1 / real_t(2*M_PI);
  real_t WAVEK = 20*M_PI;
  complex_t I = std::complex<real_t>(0., 1.);
#pragma omp parallel for
  for(int i=0; i<trg_coord.size()/3; ++i) {
    complex_t p = 0;
    real_t * tX = &trg_coord[3*i];
    for(int j=0; j<src_value.size(); ++j) {
      vec3 dX;
      real_t * sX = &src_coord[3*j];
      for(int d=0; d<3; ++d) dX[d] = tX[d] - sX[d];
      real_t R2 = norm(dX);
      if (R2 != 0) {
        real_t R = std::sqrt(R2);
        complex_t pij = std::exp(I * WAVEK * R) * src_value[j] / R;
        p += pij;
      }
    }
    trg_value[i] += p / (4*M_PI); 
  }
}

int main() {
  // WAVEK = complex_t(1, .1) / real_t(2*M_PI);
  // Args args(argc, argv);
  struct timeval tic, toc;
  // size_t N = args.numBodies;
  size_t N = 10000;
  // double a = 0.0;
  // std::complex<double> b = 0.0;
  // std::cout << sizeof(a) << "    " << sizeof(b) << std::endl;
  // NCRIT = args.ncrit;
  srand48(0);
  RealVec src_coord, trg_coord, test_coord;
#if HELMHOLTZ
  MU = 20;
  ComplexVec src_value, trg_value, test_value;
  for(size_t i=0; i<N; i++){
    src_value.push_back(std::complex<real_t>(drand48()-0.5, drand48()-0.5));
    trg_value.push_back(std::complex<real_t>(0, 0));
  } 
#else
  RealVec src_value, trg_value, test_value;
  for(size_t i=0; i<N; i++){
    src_value.push_back(drand48()-0.5);
    trg_value.push_back(0);
  } 
#endif
  for(size_t i=0; i<3*N; i++) {
    src_coord.push_back(drand48());
    trg_coord.push_back(drand48());
  }
  for(size_t i=0; i<3*N; i++) {
    test_coord.push_back(trg_coord[i]);
  }
  for(size_t i=0; i<N; i++) {
    test_value.push_back(trg_value[i]);
    // trg2_value.push_back(trg_value[i]);
    // trg2_value.push_back(trg_value[i]);
    // trg2_value.push_back(trg_value[i]);
  }

  gettimeofday(&tic, NULL);
#if HELMHOLTZ
  helmholtz_kernel(src_coord, src_value, trg_coord, trg_value);
#else
  laplace_kernel(src_coord, src_value, trg_coord, trg_value);
#endif
  gettimeofday(&toc, NULL);
  gettimeofday(&tic, NULL);
  potential_P2P(src_coord, src_value, test_coord, test_value);
  gettimeofday(&toc, NULL);
  double diff = 0;
  for(int i = 0; i < N; i++) {
#if HELMHOLTZ
    diff += test_value[i].real() - trg_value[i].real();
    diff += test_value[i].imag() - trg_value[i].imag();
#else
    diff += test_value[i] - trg_value[i];
#endif
  }
  std::cout << "diff is: " << diff/N << std::endl;
  return 0;
}
