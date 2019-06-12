#if HELMHOLTZ
#include "helmholtz.h"
#include "precompute_helmholtz.h"
#else
#include "laplace.h"
#include "precompute_laplace.h"
#endif
#include "exafmm_t.h"
#include "timer.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  vec3 X0;
  real_t R0;
#if HELMHOLTZ
  real_t WAVEK;
#endif
}

using namespace exafmm_t;

#if HELMHOLTZ
void helmholtz_kernel(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
  complex_t I = std::complex<real_t>(0., 1.);
#pragma omp parallel for
  for(size_t i=0; i<trg_coord.size()/3; ++i) {
    complex_t p = 0;
    cvec3 F = complex_t(0,0);
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
        for(int d=0; d<3; ++d) {
          F[d] += (1/R2 - WAVEK*I/R) * pij * dX[d];
        }
      }
    }
    trg_value[4*i+0] += p / (4*PI);
    trg_value[4*i+1] += F[0] / (4*PI);
    trg_value[4*i+2] += F[1] / (4*PI);
    trg_value[4*i+3] += F[2] / (4*PI);
  }
}
#else
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
    trg_value[4*t] = p / (4 * PI);
    trg_value[4*t+1] = tx / (-4 * PI);
    trg_value[4*t+2] = ty / (-4 * PI);
    trg_value[4*t+3] = tz / (-4 * PI);
  }
}
#endif

int main(int argc, char **argv) {
  size_t N = 20000;
  srand48(0);
  // initialize coordinates
  RealVec src_coord, trg_coord;
  for(size_t i=0; i<3*N; ++i) {
    src_coord.push_back(drand48());
    trg_coord.push_back(drand48());
  }
  // initialize charges and potentials
#if HELMHOLTZ
  WAVEK = 20;
  ComplexVec src_value, trg_value(4*N, complex_t(0,0)), test_value(4*N, complex_t(0,0));
  for(size_t i=0; i<N; ++i){
    src_value.push_back(complex_t(drand48()-0.5, drand48()-0.5));
  } 
#else
  RealVec src_value, trg_value(4*N, 0), test_value(4*N, 0);
  for(size_t i=0; i<N; i++){
    src_value.push_back(drand48()-0.5);
  } 
#endif

  start("non-SIMD P2P");
#if HELMHOLTZ
  helmholtz_kernel(src_coord, src_value, trg_coord, trg_value);
#else
  laplace_kernel(src_coord, src_value, trg_coord, trg_value);
#endif
  stop("non-SIMD P2P");

  start("SIMD P2P Time");
  gradient_P2P(src_coord, src_value, trg_coord, test_value);
  stop("SIMD P2P Time");

  // calculate error
  double p_diff = 0, p_norm = 0, F_diff = 0, F_norm = 0;
#pragma omp parallel for
  for(size_t i=0; i<N; ++i) {
    p_norm += std::norm(trg_value[4*i]);
    p_diff += std::norm(trg_value[4*i]-test_value[4*i]);
    for(int d=1; d<4; ++d) {
      F_norm += std::norm(trg_value[4*i+d]);
      F_diff += std::norm(trg_value[4*i+d]-test_value[4*i+d]);
    }
  }
  double p_err = sqrt(p_diff/p_norm);
  double F_err = sqrt(F_diff/F_norm);
  print("Potential Error", p_err);
  print("Gradient Error", F_err);
  double threshold = (sizeof(real_t)==4) ? 1e-5 : 1e-10;
  assert(p_err < threshold);
  assert(F_err < threshold);
  return 0;
}
