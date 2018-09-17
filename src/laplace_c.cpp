#include "laplace_c.h"

namespace exafmm_t {
  int MULTIPOLE_ORDER;
  int NSURF;
  int MAXLEVEL;
  M2LData M2Ldata;

  //! using blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
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

  RealVec transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void potentialP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEF = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = sx - tx;
        simdvec sy(src_coord[3*s+1]);
        sy = sy - ty;
        simdvec sz(src_coord[3*s+2]);
        sz = sz - tz;
        simdvec sv(src_value[s]);
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;
        tv += invR * sv;
      }
      tv *= coef;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[t+k] += tv[k];
      }
    }
    //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*20);
  }

  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    for(int i=0; i<trg_value.size(); ++i) {
      complex_t p = 0;
      real_t * tX = &trg_coord[3*i];
      for(int j=0; j<src_value.size(); ++j) {
        vec3 dX;
        real_t * sX = &src_coord[3*j];
        for(int d=0; d<3; ++d) dX[d] = tX[d] - sX[d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          complex_t invR = src_value[j] * sqrt(1.0/R2);
          p += invR;
        }
      }
      trg_value[i] += p;
    }
  }

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    for(int i=0; i<trg_value.size(); ++i) {
      complex_t p = 0;
      cvec3 F = complex_t(0., 0.);
      real_t * tX = &trg_coord[3*i];
      for(int j=0; j<src_value.size(); ++j) {
        vec3 dX;
        real_t * sX = &src_coord[3*j];
        for(int d=0; d<3; ++d) dX[d] = tX[d] - sX[d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          real_t invR2 = 1.0 / R2;
          complex_t invR = src_value[j] * sqrt(invR2);
          p += invR;
          for(int d=0; d<3; ++d) F[d] += dX[d] * invR2 * invR;
        }
      }
      trg_value[4*i+0] += p;
      trg_value[4*i+1] += F[0];
      trg_value[4*i+2] += F[1];
      trg_value[4*i+3] += F[2];
    }
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  // For Laplace: ker_dim[0] = 1, j = 0; Force a unit charge (q=1)
  // r_src layout: [x1, y1, z1, x2, y2, z2, ...]
  // k_out layout (potential): [p11, p12, p13, ..., p21, p22, ...]  (1st digit: src_idx; 2nd: trg_idx)
  // k_out layout (gradient) : [Fx11, Fy11, Fz11, Fx12, Fy12, Fz13, ... Fx1n, Fy1n, Fz1n, ...
  //                            Fx21, Fy21, Fz21, Fx22, Fy22, Fz22, ... Fx2n, Fy2n, Fz2n, ...
  //                            ...]
  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1.);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0.);
      potentialP2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }
}//end namespace
