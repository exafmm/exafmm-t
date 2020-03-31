#include <algorithm>    // std::generate
#include <type_traits>  // std::is_same
#include "exafmm_t.h"
#include "modified_helmholtz.h"
#include "timer.h"

using namespace exafmm_t;
real_t WAVEK;

void modified_helmholtz_kernel(RealVec& src_coord, RealVec& src_value,
                               RealVec& trg_coord, RealVec& trg_value) {
  int nsrcs = src_coord.size() / 3;
  int ntrgs = trg_coord.size() / 3;
  for (int i=0; i<ntrgs; ++i) {
    real_t potential = 0;
    vec3 gradient = 0;
    for (int j=0; j<nsrcs; ++j) {
      vec3 dx;
      for (int d=0; d<3; ++d) {
        dx[d] = trg_coord[3*i+d] - src_coord[3*j+d];
      }
      real_t r2 = norm(dx);
      if (r2>0) {
        real_t r = std::sqrt(r2);
        real_t kernel = std::exp(-WAVEK*r) / r * src_value[j];
        real_t dpdr = - kernel * (WAVEK*r+1) / r / r;
        potential += kernel;
        gradient[0] += dpdr * dx[0];
        gradient[1] += dpdr * dx[1];
        gradient[2] += dpdr * dx[2];
      }
    }
    trg_value[4*i+0] += potential / (4*PI);
    trg_value[4*i+1] += gradient[0] / (4*PI);
    trg_value[4*i+2] += gradient[1] / (4*PI);
    trg_value[4*i+3] += gradient[2] / (4*PI);
  }
}

int main(int argc, char **argv) {
  Args args(argc, argv);
  int n = 10001;
  std::srand(0);

  ModifiedHelmholtzFmm fmm;
  WAVEK = fmm.wavek;
  int nthreads = args.threads;
  omp_set_num_threads(nthreads);

  // initialize sources and targets
  RealVec src_coord(3*n);
  RealVec trg_coord(3*n);
  RealVec src_value(n);
  RealVec trg_value(4*n, 0);        // non-simd result
  RealVec trg_value_simd(4*n, 0);   // simd result
  std::generate(src_coord.begin(), src_coord.end(), std::rand);
  std::generate(trg_coord.begin(), trg_coord.end(), std::rand);
  std::generate(src_value.begin(), src_value.end(), std::rand);

  // direct summation
  start("non-SIMD P2P Time");
  modified_helmholtz_kernel(src_coord, src_value, trg_coord, trg_value);
  stop("non-SIMD P2P Time");

  start("SIMD P2P Time");
  fmm.gradient_P2P(src_coord, src_value, trg_coord, trg_value_simd);
  stop("SIMD P2P Time");

  // calculate error
  double p_diff = 0, p_norm = 0;   // potential
  double g_diff = 0, g_norm = 0;   // gradient
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
