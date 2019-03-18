#include <sys/time.h>
#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#else
#include "laplace.h"
#include "precompute_laplace.h"
#endif
#include "exafmm_t.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 XMIN0;
  real_t R0;
#if HELMHOLTZ
  real_t MU;
#endif
}

using namespace exafmm_t;
using namespace std;

void laplace_kernel(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  int trg_cnt = trg_coord.size()/3;
#pragma omp parallel for
  for(int t = 0; t < trg_cnt; t++) {
    real_t p = 0, tx = 0, ty = 0, tz = 0;
    for(size_t s = 0; s < src_value.size(); s++) {
      real_t r = 0;
      for(int k = 0; k < 3; k++) {
        r += (trg_coord[t*3+k] - src_coord[s*3+k])*(trg_coord[t*3+k] - src_coord[s*3+k]);
      }
      r = sqrt(r);
      if(r != 0) {
        p += src_value[s]/r;
        tx += src_value[s] * (trg_coord[t*3] - src_coord[s*3])/(r * r * r);
        ty += src_value[s] * (trg_coord[t*3+1] - src_coord[s*3+1])/(r * r * r);
        tz += src_value[s] * (trg_coord[t*3+2] - src_coord[s*3+2])/(r * r * r);
      }
    }
    trg_value[4*t] = p / (4 * M_PI);
    trg_value[4*t+1] = tx / (-4 * M_PI);
    trg_value[4*t+2] = ty / (-4 * M_PI);
    trg_value[4*t+3] = tz / (-4 * M_PI);
  }
}

void helmholtz_kernel(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
  // complex_t WAVEK = 0.1 / real_t(2*M_PI);
  real_t WAVEK = 20*M_PI;
  complex_t I = std::complex<real_t>(0., 1.);
#pragma omp parallel for
  for(size_t i=0; i<trg_coord.size()/3; ++i) {
    complex_t p = 0;
    real_t * tX = &trg_coord[3*i];
    for(size_t j=0; j<src_value.size(); ++j) {
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

int main(int argc, char **argv) {
  // WAVEK = complex_t(1, .1) / real_t(2*M_PI);
  Args args(argc, argv);
  struct timeval tic, toc;
  size_t N = args.numBodies;
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
    trg_value.push_back(0);
    trg_value.push_back(0);
    trg_value.push_back(0);
  } 
#endif
  std::cout << std::setw(20) << std::left << "Number of bodies" << " : " << std::scientific << N << std::endl;
  for(size_t i=0; i<3*N; i++) {
    src_coord.push_back(drand48());
    trg_coord.push_back(drand48());
  }
  for(size_t i=0; i<3*N; i++) {
    test_coord.push_back(trg_coord[i]);
  }
  for(size_t i=0; i<4*N; i++) {
    test_value.push_back(trg_value[i]);
  }

  gettimeofday(&tic, NULL);
#if HELMHOLTZ
  helmholtz_kernel(src_coord, src_value, trg_coord, trg_value);
#else
  laplace_kernel(src_coord, src_value, trg_coord, trg_value);
#endif
  gettimeofday(&toc, NULL);
  gettimeofday(&tic, NULL);
  gradient_P2P(src_coord, src_value, test_coord, test_value);
  gettimeofday(&toc, NULL);
  double p_diff = 0, g_diff = 0;
#pragma omp parallel for
  for(size_t i = 0; i < N; i++) {
#if HELMHOLTZ
    p_diff += test_value[4*i].real() - trg_value[4*i].real();
    g_diff += test_value[4*i].imag() - trg_value[4*i].imag();
#else
    p_diff += test_value[4*i] - trg_value[4*i];
    g_diff += test_value[4*i+1] - trg_value[4*i+1];
    g_diff += test_value[4*i+2] - trg_value[4*i+2];
    g_diff += test_value[4*i+3] - trg_value[4*i+3];
#endif
  }
  std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << p_diff/N << std::endl;
  std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << g_diff/N << std::endl;
  return 0;
}
