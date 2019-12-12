#include <algorithm>  // std::generate
#include "laplace.h"
#include "exafmm_t.h"
#include "timer.h"
#include <omp.h>

using namespace exafmm_t;

void laplace_kernel(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  int ntrgs = trg_coord.size() / 3;
#pragma omp parallel for
  for (int t=0; t<ntrgs; t++) {
    real_t potential = 0;
    vec3 gradient = 0;
    for (size_t s=0; s<src_value.size(); ++s) {
      vec3 dx = 0;
      for (int d=0; d<3; ++d) {
        dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
      }
      real_t r2 = norm(dx);
      if (r2!=0) {
        real_t inv_r2 = 1.0 / r2;
        real_t inv_r = src_value[s] * std::sqrt(inv_r2);
        potential += inv_r;
        dx *= inv_r2 * inv_r;
        gradient[0] += dx[0];
        gradient[1] += dx[1];
        gradient[2] += dx[2];
      }
    }
    trg_value[4*t] += potential / (4*PI);
    trg_value[4*t+1] += gradient[0] / (-4*PI);
    trg_value[4*t+2] += gradient[1] / (-4*PI);
    trg_value[4*t+3] += gradient[2] / (-4*PI);
  }
}

int main(int argc, char **argv) {
  Args args(argc, argv);
  int n = 10000;
  std::srand(0);
  LaplaceFMM fmm;

  int nthreads = args.threads;
  omp_set_num_threads(nthreads);

  // initialize sources and targets
  RealVec src_coord(3*n);
  RealVec trg_coord(3*n);
  RealVec src_value(n);
  RealVec trg_value(4*n, 0);
  std::generate(src_coord.begin(), src_coord.end(), std::rand);
  std::generate(trg_coord.begin(), trg_coord.end(), std::rand);
  std::generate(src_value.begin(), src_value.end(), std::rand);

  start("non-SIMD P2P Time");
  laplace_kernel(src_coord, src_value, trg_coord, trg_value);
  stop("non-SIMD P2P Time");

  // chunk targets to parallelize gradient P2P
  std::vector<RealVec> trg_coord_(nthreads);
  std::vector<RealVec> trg_value_(nthreads);
  int ntrgs = n / nthreads;
  for (int i=0; i<nthreads; ++i) {
    int trg_begin = i*ntrgs;
    int trg_end = std::min((i+1)*ntrgs, n);
    trg_coord_[i] = RealVec(trg_coord.begin()+3*trg_begin, trg_coord.begin()+3*trg_end);
    trg_value_[i] = RealVec(4*(trg_end-trg_begin), 0); 
  }

  start("SIMD P2P Time");
#pragma omp parallel for
  for (int i=0; i<nthreads; ++i) {
    fmm.gradient_P2P(src_coord, src_value, trg_coord_[i], trg_value_[i]);
  }
  stop("SIMD P2P Time");

  // collect simd target values from chunks
  RealVec trg_value_simd;
  for (int i=0; i<nthreads; ++i) {
    trg_value_simd.insert(trg_value_simd.end(), trg_value_[i].begin(), trg_value_[i].end());
  }

  // calculate error
  double p_diff = 0, p_norm = 0, F_diff = 0, F_norm = 0;
#pragma omp parallel for
  for(int i=0; i<n; ++i) {
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
