#include <algorithm>  // std::generate
#include "laplace.h"
#include "exafmm_t.h"
#include "timer.h"

using namespace exafmm_t;

void laplace_kernel(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  int trg_cnt = trg_coord.size() / 3;
#pragma omp parallel for
  for (int t=0; t<trg_cnt; t++) {
    real_t potential = 0;
    vec3 gradient = 0;
    for (size_t s=0; s<src_value.size(); ++s) {
      real_t r = 0;
      for (int d=0; d<3; ++d) {
        r += (trg_coord[t*3+d]-src_coord[s*3+d]) * (trg_coord[t*3+d]-src_coord[s*3+d]);
      }
      r = std::sqrt(r);
      if (r!=0) {
        potential += src_value[s] / r;
        gradient[0] += src_value[s] * (trg_coord[t*3+0] - src_coord[s*3+0]) / (r*r*r);
        gradient[1] += src_value[s] * (trg_coord[t*3+1] - src_coord[s*3+1]) / (r*r*r);
        gradient[2] += src_value[s] * (trg_coord[t*3+2] - src_coord[s*3+2]) / (r*r*r);
      }
    }
    trg_value[4*t] += potential / (4*PI);
    trg_value[4*t+1] += gradient[0] / (-4*PI);
    trg_value[4*t+2] += gradient[1] / (-4*PI);
    trg_value[4*t+3] += gradient[2] / (-4*PI);
  }
}

int main(int argc, char **argv) {
  size_t n = 10000;
  std::srand(0);
  LaplaceFMM fmm;

  // initialize sources and targets
  RealVec src_coord(3*n);
  RealVec trg_coord(3*n);
  RealVec src_value(n);
  RealVec trg_value(4*n, 0);
  RealVec trg_value_simd(4*n, 0);
  std::generate(src_coord.begin(), src_coord.end(), std::rand);
  std::generate(trg_coord.begin(), trg_coord.end(), std::rand);
  std::generate(src_value.begin(), src_value.end(), std::rand);

  start("non-SIMD P2P Time");
  laplace_kernel(src_coord, src_value, trg_coord, trg_value);
  stop("non-SIMD P2P Time");

  start("SIMD P2P Time");
  fmm.gradient_P2P(src_coord, src_value, trg_coord, trg_value_simd);
  stop("SIMD P2P Time");

  // calculate error
  double p_diff = 0, p_norm = 0, F_diff = 0, F_norm = 0;
#pragma omp parallel for
  for(size_t i=0; i<n; ++i) {
    p_norm += std::norm(trg_value[4*i]);
    p_diff += std::norm(trg_value[4*i]-trg_value_simd[4*i]);
    for(int d=1; d<4; ++d) {
      F_norm += std::norm(trg_value[4*i+d]);
      F_diff += std::norm(trg_value[4*i+d]-trg_value_simd[4*i+d]);
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
