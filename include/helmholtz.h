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
    HelmholtzFmm(int p_, int ncrit_, int depth_, real_t wavek_, std::string filename_=std::string()) :
      Fmm<complex_t>(p_, ncrit_, depth_, filename_)
    {
      wavek = wavek_;
      if (this->filename.empty()) {
        this->filename = std::string("helmholtz_") + (std::is_same<real_t, float>::value ? "f" : "d")
                       + std::string("_p") + std::to_string(p) + std::string(".dat");
      }
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
      real_t newton_coef = 16;
      simdvec coef(real_t(1.0/(4*PI*newton_coef)));
      simdvec k(wavek/newton_coef);
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
          sx -= tx;
          simdvec sy(src_coord[3*s+1]);
          sy -= ty;
          simdvec sz(src_coord[3*s+2]);
          sz -= tz;
          simdvec sv_real(src_value[s].real());
          simdvec sv_imag(src_value[s].imag());
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invr = rsqrt(r2);   // invr = newton_coef * 1/r
          invr &= r2 > zero;

          simdvec kr = k * r2 * invr;   // newton_coefs in k & invr cancel out
          simdvec G_real = cos(kr) * invr;  // G = e^(ikr) / r
          simdvec G_imag = sin(kr) * invr;  // invr carries newton_coef
          tv_real += sv_real*G_real - sv_imag*G_imag;  // p += G * q
          tv_imag += sv_real*G_imag + sv_imag*G_real;
        }
        tv_real *= coef;  // coef carries 1/(4*PI) and offsets newton_coef in invr
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
      real_t newton_coef = 16;   // comes from Newton's method in simd rsqrt function
      simdvec coef(real_t(1.0/(4*PI*newton_coef)));
      simdvec k(wavek/newton_coef);
      simdvec newton_offset(real_t(1.0/(newton_coef*newton_coef)));   // offset invr2 term
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      const complex_t I(0, 1);
      for (t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv0_real(zero);
        simdvec tv0_imag(zero);
        simdvec tv1_real(zero);
        simdvec tv1_imag(zero);
        simdvec tv2_real(zero);
        simdvec tv2_imag(zero);
        simdvec tv3_real(zero);
        simdvec tv3_imag(zero);
        for (int s=0; s<nsrcs; s++) {
          simdvec sx(src_coord[3*s+0]);
          sx -= tx;    // sx is negative dx
          simdvec sy(src_coord[3*s+1]);
          sy -= ty;
          simdvec sz(src_coord[3*s+2]);
          sz -= tz;
          simdvec sv_real(src_value[s].real());
          simdvec sv_imag(src_value[s].imag());
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invr = rsqrt(r2);
          invr &= r2 > zero;

          simdvec kr = k * r2 * invr;   // newton_coefs in k & invr cancel out
          simdvec G_real = cos(kr) * invr;  // G = e^(ikr) / r
          simdvec G_imag = sin(kr) * invr;  // invr carries newton_coef
          simdvec potential_real = sv_real*G_real - sv_imag*G_imag;    // p = G * q
          simdvec potential_imag = sv_real*G_imag + sv_imag*G_real;
          tv0_real += potential_real;
          tv0_imag += potential_imag;
          // coefg = 1/r2 - k/r*I (considering the negative sign in sx)
          simdvec coefg_real = invr * invr * newton_offset;
          simdvec coefg_imag = - k * invr;
          // gradient = coefg * potential * dx
          simdvec gradient_real = coefg_real*potential_real - coefg_imag*potential_imag;
          simdvec gradient_imag = coefg_real*potential_imag + coefg_imag*potential_real;
          tv1_real += sx * gradient_real;
          tv1_imag += sx * gradient_imag;
          tv2_real += sy * gradient_real;
          tv2_imag += sy * gradient_imag;
          tv3_real += sz * gradient_real;
          tv3_imag += sz * gradient_imag;
        }
        tv0_real *= coef;
        tv0_imag *= coef;
        tv1_real *= coef;
        tv1_imag *= coef;
        tv2_real *= coef;
        tv2_imag *= coef;
        tv3_real *= coef;
        tv3_imag *= coef;
        for (int m=0; m<NSIMD && (t+m)<ntrgs; m++) {
          trg_value[4*(t+m)+0] += complex_t(tv0_real[m], tv0_imag[m]);
          trg_value[4*(t+m)+1] += complex_t(tv1_real[m], tv1_imag[m]);
          trg_value[4*(t+m)+2] += complex_t(tv2_real[m], tv2_imag[m]);
          trg_value[4*(t+m)+3] += complex_t(tv3_real[m], tv3_imag[m]);
        }
      }
      for (; t<ntrgs; t++) {
        complex_t potential(0, 0);
        cvec3 gradient = complex_t(0., 0.);
        for (int s=0; s<nsrcs; s++) {
          vec3 dx;
          for (int d=0; d<3; d++)
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          real_t r2 = norm(dx);
          if (r2 != 0) {
            real_t r = std::sqrt(r2);
            complex_t potential_ij = std::exp(I * r * wavek) * src_value[s] / r;
            complex_t coefg = (I*wavek/r - 1/r2) * potential_ij;
            potential += potential_ij;
            for (int d=0; d<3; d++)
              gradient[d] += coefg * dx[d];
          }
        }
        trg_value[4*t+0] += potential / (4*PI);
        trg_value[4*t+1] += gradient[0] / (4*PI);
        trg_value[4*t+2] += gradient[1] / (4*PI);
        trg_value[4*t+3] += gradient[2] / (4*PI);
      }
    }
  };
}  // end namespace exafmm_t
#endif
