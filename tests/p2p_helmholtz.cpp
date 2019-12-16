#include <algorithm>  // std::generate
#include "helmholtz.h"
#include "exafmm_t.h"
#include "timer.h"

using namespace exafmm_t;
real_t WAVEK;

void helmholtz_kernel(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
  complex_t I = std::complex<real_t>(0., 1.);
#pragma omp parallel for
  for (size_t i=0; i<trg_coord.size()/3; ++i) {
    complex_t potential = 0;
    cvec3 gradient = complex_t(0,0);
    for (size_t j=0; j<src_value.size(); ++j) {
      vec3 dx;
      for (int d=0; d<3; ++d) {
        dx[d] = trg_coord[3*i+d] - src_coord[3*j+d];
      }
      real_t r2 = norm(dx);
      if (r2!=0) {
        real_t r = std::sqrt(r2);
        complex_t pij = std::exp(I * WAVEK * r) * src_value[j] / r;
        potential += pij;
        for (int d=0; d<3; ++d) {
          gradient[d] += (1/r2 - WAVEK*I/r) * pij * dx[d];
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
  int n = 10000;
  std::srand(0);
  HelmholtzFMM fmm;
  WAVEK = fmm.wavek;
  int nthreads = args.threads;
  omp_set_num_threads(nthreads);

  // initialize sources and targets
  RealVec src_coord(3*n);
  RealVec trg_coord(3*n);
  ComplexVec src_value(n);
  ComplexVec trg_value(4*n, 0);
  std::generate(src_coord.begin(), src_coord.end(), std::rand);
  std::generate(trg_coord.begin(), trg_coord.end(), std::rand);
  std::generate(src_value.begin(), src_value.end(), []() {
                  return complex_t(std::rand(), std::rand());
                });

  start("non-SIMD P2P");
  helmholtz_kernel(src_coord, src_value, trg_coord, trg_value);
  stop("non-SIMD P2P");

  // chunk targets to parallelize gradient P2P
  std::vector<RealVec> trg_coord_(nthreads);
  std::vector<ComplexVec> trg_value_(nthreads);
  int ntrgs = n / nthreads;
  for (int i=0; i<nthreads; ++i) {
    int trg_begin = i*ntrgs;
    int trg_end = std::min((i+1)*ntrgs, n);
    trg_coord_[i] = RealVec(trg_coord.begin()+3*trg_begin, trg_coord.begin()+3*trg_end);
    trg_value_[i] = ComplexVec(4*(trg_end-trg_begin), 0.); 
  }

  start("SIMD P2P Time");
#pragma omp parallel for
  for (int i=0; i<nthreads; ++i) {
    fmm.gradient_P2P(src_coord, src_value, trg_coord_[i], trg_value_[i]);
  }
  stop("SIMD P2P Time");

  // collect simd target values from chunks
  ComplexVec trg_value_simd;
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
