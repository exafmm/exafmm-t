#include "helmholtz.h"

namespace <{1:exafmm_t}> {
  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    int newton_scale = 1;
    for(int i=0; i<2; i++) {
      newton_scale = 2*newton_scale*newton_scale*newton_scale;
    }
    const real_t COEF = 1.0/(newton_scale*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    simdvec mu(20.0*M_PI/newton_scale);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv_real(zero);
      simdvec tv_imag(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = sx - tx;
        simdvec sy(src_coord[3*s+1]);
        sy = sy - ty;
        simdvec sz(src_coord[3*s+2]);
        sz = sz - tz;
        simdvec sv_real(src_value[s].real());
        simdvec sv_imag(src_value[s].imag());
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;

        simdvec mu_r = mu * r2 * invR;
        simdvec G0 = cos(mu_r)*invR;
        simdvec G1 = sin(mu_r)*invR;
        tv_real += sv_real*G0 - sv_imag*G1;
        tv_imag += sv_real*G1 + sv_imag*G0;
      }
      tv_real *= coef;
      tv_imag *= coef;
      for(int k=0; k<NSIMD && (t+k)<trg_cnt; k++) {
      std::cout << "P2P_simd: " << tv_real[k] << " , " << tv_imag[k] << std::endl;
        trg_value[t+k] += complex_t(tv_real[k], tv_imag[k]);
      }
    }
  }
}
