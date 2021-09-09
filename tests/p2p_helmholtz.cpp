#include <algorithm>
#include <random>
#include <type_traits>
#include "exafmm_t.h"
#include "helmholtz.h"
#include "timer.h"

using namespace exafmm_t;

void helmholtz_kernel(RealVec& src_coord, ComplexVec& src_value,
                      RealVec& trg_coord, ComplexVec& trg_value, complex_t wavek) {
  complex_t I = std::complex<real_t>(0., 1.);
  int nsrcs = src_coord.size() / 3;
  int ntrgs = trg_coord.size() / 3;
  for (int i=0; i<ntrgs; ++i) {
    complex_t potential = 0;
    cvec3 gradient = complex_t(0,0);
    for (int j=0; j<nsrcs; ++j) {
      vec3 dx;
      for (int d=0; d<3; ++d) {
        dx[d] = trg_coord[3*i+d] - src_coord[3*j+d];
      }
      real_t r2 = norm(dx);
      if (r2!=0) {
        real_t r = std::sqrt(r2);
        complex_t potential_ij = std::exp(I * wavek * r) * src_value[j] / r;
        potential += potential_ij;
        for (int d=0; d<3; ++d) {
          gradient[d] += (wavek*I/r - 1/r2) * potential_ij * dx[d];
        }
      }
    }
    trg_value[4*i] += potential / (4*PI);
    trg_value[4*i+1] += gradient[0] / (4*PI);
    trg_value[4*i+2] += gradient[1] / (4*PI);
    trg_value[4*i+3] += gradient[2] / (4*PI);
  }
}

int main(int argc, char **argv) {
  Args args(argc, argv);
  int n_max = 20001;
  int n = std::min(args.numBodies, n_max);

  HelmholtzFmm fmm;
  fmm.wavek = complex_t(5.,10.);

  // initialize sources and targets
  print("numBodies", n);
  RealVec src_coord(3*n);
  RealVec trg_coord(3*n);
  ComplexVec src_value(n);
  ComplexVec trg_value(4*n, 0);
  ComplexVec trg_value_simd(4*n, 0);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<real_t> dist(-1.0, 1.0);
  auto gen = [&dist, &engine]() {
    return dist(engine);
  };

  std::generate(src_coord.begin(), src_coord.end(), gen);
  std::generate(trg_coord.begin(), trg_coord.end(), gen);
  std::generate(src_value.begin(), src_value.end(), [&gen]() {
                  return complex_t(gen(), gen());
                });

  // direct summation
  start("non-SIMD P2P");
  helmholtz_kernel(src_coord, src_value, trg_coord, trg_value, fmm.wavek);
  stop("non-SIMD P2P");

  start("SIMD P2P Time");
  fmm.gradient_P2P(src_coord, src_value, trg_coord, trg_value_simd);
  stop("SIMD P2P Time");

  // calculate error
  double p_diff = 0, p_norm = 0;
  double g_diff = 0, g_norm = 0;
  for (int i=0; i<n; ++i) {
    p_norm += std::norm(trg_value[4*i]);
    p_diff += std::norm(trg_value[4*i]-trg_value_simd[4*i]);
    for (int d=1; d<4; ++d) {
      g_norm += std::norm(trg_value[4*i+d]);
      g_diff += std::norm(trg_value[4*i+d]-trg_value_simd[4*i+d]);
    }
  }
  double p_err = sqrt(p_diff/p_norm);
  double g_err = sqrt(g_diff/g_norm);
  print("Potential Error", p_err);
  print("Gradient Error", g_err);

  double threshold = std::is_same<float, real_t>::value ? 1e-6 : 1e-12;
  assert(p_err < threshold);
  assert(g_err < threshold);

  return 0;
}
