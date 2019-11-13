#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "modified_helmholtz.h"

namespace exafmm_t {
  //! blas gemv with row major data
  void ModifiedHelmholtzFMM::gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
    char trans = 'T';
    real_t alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
#if FLOAT
    sgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    dgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }

  //! blas gemm with row major data
  void ModifiedHelmholtzFMM::gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void ModifiedHelmholtzFMM::svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    RealVec WORK(LWORK);
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  RealVec ModifiedHelmholtzFMM::transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void ModifiedHelmholtzFMM::potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    for (int i=0; i<ntrgs; ++i) {
      vec3 x_trg;
      for (int d=0; d<3; ++d)
        x_trg[d] = trg_coord[3*i+d];
      for (int j=0; j<nsrcs; ++j) {
        vec3 x_src;
        for (int d=0; d<3; ++d)
          x_src[d] = src_coord[3*j+d];
        vec3 dx = x_src - x_trg;
        real_t r = std::sqrt(norm(dx));
        if (r>0)
          trg_value[i] += std::exp(-wavek*r) / r;
      }
      trg_value[i] /= 4*PI;
    }
  }

  //! P2P save pairwise contributions to k_out (not aggregate over each target)
  void ModifiedHelmholtzFMM::kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1.);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0.);
      potential_P2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }
}  // end namespace exafmm_t
