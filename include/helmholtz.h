#ifndef helmholtz_h
#define helmholtz_h
#include "exafmm_t.h"
#include "fmm.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

namespace exafmm_t {
  //! A derived FMM class for Helmholtz kernel.
  class HelmholtzFmm : public Fmm<complex_t> {
  public:
    real_t wavek;      //!< Wave number k.

    HelmholtzFmm() {}
    HelmholtzFmm(int p_, int ncrit_, int depth_, real_t wavek_) : Fmm(p_, ncrit_, depth_) {
      wavek = wavek_;
      this->filename = std::string("helmholtz_") + (std::is_same<real_t, float>::value ? "f" : "d")
                     + std::string("_p") + std::to_string(p) + std::string(".dat");
    }

    /**
     * @brief Compute potentials at targets induced by sources directly.
     * 
     * @param src_coord Vector of coordinates of sources.
     * @param src_value Vector of charges of sources.
     * @param trg_coord Vector of coordinates of targets.
     * @param trg_value Vector of potentials of targets.
     */
    void potential_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
      simdvec zero((real_t)0);
      real_t newton_scale = 16;   // comes from Newton's method in simd rsqrt function
      const real_t COEF = 1.0/(4*PI*newton_scale);
      simdvec coef(COEF);
      simdvec k(wavek/newton_scale);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      const complex_t I(0, 1);
      for (t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv_real(zero);
        simdvec tv_imag(zero);
        for (int s=0; s<nsrcs; s++) {
          simdvec sx(src_coord[3*s+0]);
          sx = tx - sx;
          simdvec sy(src_coord[3*s+1]);
          sy = ty - sy;
          simdvec sz(src_coord[3*s+2]);
          sz = tz - sz;
          simdvec sv_real(src_value[s].real());
          simdvec sv_imag(src_value[s].imag());
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invR = rsqrt(r2);   // invR = newton_scale * 1/r
          invR &= r2 > zero;

          simdvec kr = k * r2 * invR;   // newton_scales in k & invR cancel out
          simdvec G_real = cos(kr) * invR;  // G = e^(ikr) / r
          simdvec G_imag = sin(kr) * invR;  // invR carries newton_scale
          tv_real += sv_real*G_real - sv_imag*G_imag;  // p += G * q
          tv_imag += sv_real*G_imag + sv_imag*G_real;
        }
        tv_real *= coef;  // coef carries 1/(4*PI) and offsets newton_scale in invR
        tv_imag *= coef;
        for (int m=0; m<NSIMD && (t+m)<ntrgs; m++) {
          trg_value[t+m] += complex_t(tv_real[m], tv_imag[m]);
        }
      }
      for (; t<ntrgs; t++) {
        complex_t potential(0, 0);
        for (int s=0; s<nsrcs; s++) {
          vec3 dx;
          for (int d=0; d<3; d++)
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          real_t r2 = norm(dx);
          if (r2 != 0) {
            real_t r = std::sqrt(r2);
            potential += std::exp(I * r * wavek) * src_value[s] / r;
          }
        }
        trg_value[t] += potential / (4*PI);
      }
    }

    /**
     * @brief Compute potentials and gradients at targets induced by sources directly.
     * 
     * @param src_coord Vector of coordinates of sources.
     * @param src_value Vector of charges of sources.
     * @param trg_coord Vector of coordinates of targets.
     * @param trg_value Vector of potentials of targets.
     */
    void gradient_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
      simdvec zero((real_t)0);
      real_t newton_scale = 16;   // comes from Newton's method in simd rsqrt function
      const real_t COEF = 1.0/(4*PI*newton_scale);   // factor 16 comes from the simd rsqrt function
      simdvec coef(COEF);
      simdvec k(wavek/newton_scale);
      simdvec NEWTON(newton_scale);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      const complex_t I(0, 1);
      for (t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
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
        for (int s=0; s<nsrcs; s++) {
          simdvec sx(src_coord[3*s+0]);
          sx = tx - sx;
          simdvec sy(src_coord[3*s+1]);
          sy = ty - sy;
          simdvec sz(src_coord[3*s+2]);
          sz = tz - sz;
          simdvec sv_real(src_value[s].real());
          simdvec sv_imag(src_value[s].imag());
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invR = rsqrt(r2);
          invR &= r2 > zero;

          simdvec kr = k * r2 * invR;   // newton_scales in k & invR cancel out
          simdvec G_real = cos(kr) * invR;  // G = e^(ikr) / r
          simdvec G_imag = sin(kr) * invR;  // invR carries newton_scale
          simdvec p_real = sv_real*G_real - sv_imag*G_imag;    // p = G * q
          simdvec p_imag = sv_real*G_imag + sv_imag*G_real;
          tv_real += p_real;
          tv_imag += p_imag;
          // F = -\nabla p = (1/(r^2) - k/r*I) * p * dx
          simdvec coefg_real = invR * invR / NEWTON / NEWTON;  // coefg = 1/(r^2) - k/r*I
          simdvec coefg_imag = - k * invR;
          simdvec F_real = coefg_real*p_real - coefg_imag*p_imag; // F = coefg * p * dx
          simdvec F_imag = coefg_real*p_imag + coefg_imag*p_real;
          F0_real += sx * F_real;
          F0_imag += sx * F_imag;
          F1_real += sy * F_real;
          F1_imag += sy * F_imag;
          F2_real += sz * F_real;
          F2_imag += sz * F_imag;
        }
        tv_real *= coef;
        tv_imag *= coef;
        F0_real *= coef;
        F0_imag *= coef;
        F1_real *= coef;
        F1_imag *= coef;
        F2_real *= coef;
        F2_imag *= coef;
        for (int m=0; m<NSIMD && (t+m)<ntrgs; m++) {
          trg_value[4*(t+m)+0] += complex_t(tv_real[m], tv_imag[m]);
          trg_value[4*(t+m)+1] += complex_t(F0_real[m], F0_imag[m]);
          trg_value[4*(t+m)+2] += complex_t(F1_real[m], F1_imag[m]);
          trg_value[4*(t+m)+3] += complex_t(F2_real[m], F2_imag[m]);
        }
      }
      for (; t<ntrgs; t++) {
        complex_t p(0, 0);
        cvec3 F = complex_t(0., 0.);
        for (int s=0; s<nsrcs; s++) {
          vec3 dx;
          for (int d=0; d<3; d++)
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          real_t r2 = norm(dx);
          if (r2 != 0) {
            real_t r = std::sqrt(r2);
            complex_t pij = std::exp(I * r * wavek) * src_value[s] / r;
            complex_t coefg = (1/r2 - I*wavek/r) * pij;
            p += pij;
            for (int d=0; d<3; d++)
              F[d] += coefg * dx[d];
          }
        }
        trg_value[4*t+0] += p / (4*PI);
        trg_value[4*t+1] += F[0] / (4*PI);
        trg_value[4*t+2] += F[1] / (4*PI);
        trg_value[4*t+3] += F[2] / (4*PI);
      }
    }
  };
}  // end namespace exafmm_t
#endif
