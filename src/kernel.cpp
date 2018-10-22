#include "kernel.h"

namespace exafmm_t {
  // Laplace kernel
#if EXAFMM_LAPLACE
#if COMPLEX
  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEF = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
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
        tv_real += invR * sv_real;
        tv_imag += invR * sv_imag;
      }
      tv_real *= coef;
      tv_imag *= coef;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[t+k] += complex_t(tv_real[k], tv_imag[k]);
      }
    }
  }

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEF = 1.0/(16*4*M_PI); 
    const real_t COEFG = -1.0/(16*16*16*4*M_PI); 
    simdvec coef(COEF);
    simdvec coefg(COEFG);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv_real(zero);
      simdvec tv_imag(zero);
      simdvec F0_real(zero);
      simdvec F0_imag(zero);
      simdvec F1_real(zero);
      simdvec F1_imag(zero);
      simdvec F2_real(zero);
      simdvec F2_imag(zero);
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
        tv_real += invR * sv_real;
        tv_imag += invR * sv_imag;
        F0_real += sx * invR * invR * invR * sv_real;
        F0_imag += sx * invR * invR * invR * sv_imag;
        F1_real += sy * invR * invR * invR * sv_real;
        F1_imag += sy * invR * invR * invR * sv_imag;
        F2_real += sz * invR * invR * invR * sv_real;
        F2_imag += sz * invR * invR * invR * sv_imag;
      }
      tv_real *= coef;
      tv_imag *= coef;
      F0_real *= coefg;
      F0_imag *= coefg;
      F1_real *= coefg;
      F1_imag *= coefg;
      F2_real *= coefg;
      F2_imag *= coefg;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[4*(t+k)+0] += complex_t(tv_real[k], tv_imag[k]);
        trg_value[4*(t+k)+1] += complex_t(F0_real[k], F0_imag[k]);
        trg_value[4*(t+k)+2] += complex_t(F1_real[k], F1_imag[k]);
        trg_value[4*(t+k)+3] += complex_t(F2_real[k], F2_imag[k]);
      }
    }
  }
#else
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

  void gradientP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);
    simdvec coefp(COEFP);
    simdvec coefg(COEFG);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv0(zero);
      simdvec tv1(zero);
      simdvec tv2(zero);
      simdvec tv3(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = tx - sx;
        simdvec sy(src_coord[3*s+1]);
        sy = ty - sy;
        simdvec sz(src_coord[3*s+2]);
        sz = tz - sz;
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;
        simdvec invR3 = (invR*invR) * invR;
        simdvec sv(src_value[s]);
        tv0 += sv*invR;
        sv *= invR3;
        tv1 += sv*sx;
        tv2 += sv*sy;
        tv3 += sv*sz;
      }
      tv0 *= coefp;
      tv1 *= coefg;
      tv2 *= coefg;
      tv3 *= coefg;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[0+4*(t+k)] += tv0[k];
        trg_value[1+4*(t+k)] += tv1[k];
        trg_value[2+4*(t+k)] += tv2[k];
        trg_value[3+4*(t+k)] += tv3[k];
      }
    }
    //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*27);
  }

#endif

  // Helmholtz kernel
#elif EXAFMM_HELMHOLTZ
  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    int newton_scale = 1;
    for(int i=0; i<2; i++) {
      newton_scale = 2*newton_scale*newton_scale*newton_scale;
    }
    const real_t COEF = 1.0/(newton_scale*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    simdvec mu(20*M_PI/newton_scale);
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
        trg_value[t+k] += complex_t(tv_real[k], tv_imag[k]);
      }
    }
  }

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    int newton_scale = 1;
    for(int i=0; i<2; i++) {
      newton_scale = 2*newton_scale*newton_scale*newton_scale;
    }
    const real_t COEF = 1.0/(newton_scale*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    simdvec mu(20*M_PI/newton_scale);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv_real(zero);
      simdvec tv_imag(zero);
      simdvec F0_real(zero);
      simdvec F0_imag(zero);
      simdvec F1_real(zero);
      simdvec F1_imag(zero);
      simdvec F2_real(zero);
      simdvec F2_imag(zero);
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

        simdvec mu_r = mu*r2*invR;
        simdvec G0 = cos(mu_r)*invR;
        simdvec G1 = sin(mu_r)*invR;
        simdvec p_real = sv_real*G0 - sv_imag*G1;
        simdvec p_imag = sv_real*G1 + sv_imag*G0;
        tv_real += p_real;
        tv_imag += p_imag;
        simdvec coef_real = invR*invR*p_real + mu*p_imag*invR;
        simdvec coef_imag = invR*invR*p_imag - mu*p_real*invR;
        F0_real += sx*coef_real;
        F0_imag += sx*coef_imag;
        F1_real += sy*coef_real;
        F1_imag += sy*coef_imag;
        F2_real += sz*coef_real;
        F2_imag += sz*coef_imag;
      }
      tv_real *= coef;
      tv_imag *= coef;
      for(int k=0; k<NSIMD && (t+k)<trg_cnt; k++) {
        trg_value[4*(t+k)+0] += complex_t(tv_real[k], tv_imag[k]);
        trg_value[4*(t+k)+1] += complex_t(F0_real[k], F0_imag[k]);
        trg_value[4*(t+k)+2] += complex_t(F1_real[k], F1_imag[k]);
        trg_value[4*(t+k)+3] += complex_t(F2_real[k], F2_imag[k]);
      }
    }
  }
#endif
}
