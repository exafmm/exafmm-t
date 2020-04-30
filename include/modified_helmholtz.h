#ifndef modified_helmholtz_h
#define modified_helmholtz_h
#include "exafmm_t.h"
#include "fmm.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

namespace exafmm_t {
  //! A derived FMM class for modified Helmholtz kernel.
  class ModifiedHelmholtzFmm : public Fmm<real_t> {
  public:
    real_t wavek;

    ModifiedHelmholtzFmm() {}
    ModifiedHelmholtzFmm(int p_, int ncrit_, int depth_, real_t wavek_, std::string filename_=std::string()) :
      Fmm<real_t>(p_, ncrit_, depth_, filename_)
    {
      wavek = wavek_;
      if (this->filename.empty()) {
        this->filename = std::string("modified_helmholtz_") + (std::is_same<real_t, float>::value ? "f" : "d")
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
    void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
      simdvec zero(real_t(0));
      real_t newton_coef = 16;   // it comes from Newton's method in simd rsqrt function
      simdvec coef(real_t(1.0/(4*PI*newton_coef)));
      simdvec k(-wavek/newton_coef);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      for (t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv(zero);
        for (int s=0; s<nsrcs; s++) {
          simdvec sx(src_coord[3*s+0]);
          sx -= tx;
          simdvec sy(src_coord[3*s+1]);
          sy -= ty;
          simdvec sz(src_coord[3*s+2]);
          sz -= tz;
          simdvec sv(src_value[s]);
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invr = rsqrt(r2);
          invr &= r2 > zero;

          tv += exp(k*r2*invr) * invr * sv;  // k has negative sign
        }
        tv *= coef;
        for (int m=0; m<NSIMD && (t+m)<ntrgs; m++) {
          trg_value[t+m] += tv[m];
        }
      }
      for (; t<ntrgs; t++) {
        real_t potential = 0;
        for (int s=0; s<nsrcs; s++) {
          vec3 dx;
          for (int d=0; d<3; d++)
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          real_t r2 = norm(dx);
          if (r2!=0) {
            real_t r = std::sqrt(r2);
            potential += std::exp(-wavek*r) * src_value[s] / r;
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
    void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
      simdvec zero(real_t(0));
      simdvec one(real_t(1));
      real_t newton_coef = 16;   // it comes from Newton's method in simd rsqrt function
      simdvec coefp(real_t(1.0/(4*PI*newton_coef)));  // potential: r
      simdvec coefg(real_t(1.0/(4*PI*newton_coef*newton_coef*newton_coef)));  // gradient: r3
      simdvec k(-wavek/newton_coef);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      for (t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv0(zero);
        simdvec tv1(zero);
        simdvec tv2(zero);
        simdvec tv3(zero);
        for (int s=0; s<nsrcs; s++) {
          simdvec sx(src_coord[3*s+0]);
          sx -= tx;
          simdvec sy(src_coord[3*s+1]);
          sy -= ty;
          simdvec sz(src_coord[3*s+2]);
          sz -= tz;
          simdvec sv(src_value[s]);
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invr = rsqrt(r2);
          invr &= r2 > zero;

          simdvec invr2 = invr * invr;

          simdvec kr = (k * r2) * invr;     // -k*r
          simdvec potential = exp(kr) * invr * sv;  // exp(-kr) / r * q
          tv0 += potential;
          simdvec krp1 = one - kr;          // k*r+1
          potential *= krp1 * invr2;        // exp(-kr) * (kr+1) * q / r3
          tv1 += potential * sx;
          tv2 += potential * sy;
          tv3 += potential * sz;
        }
        tv0 *= coefp;
        tv1 *= coefg;
        tv2 *= coefg;
        tv3 *= coefg;
        for (int m=0; m<NSIMD && (t+m)<ntrgs; m++) {
          trg_value[0+4*(t+m)] += tv0[m];
          trg_value[1+4*(t+m)] += tv1[m];
          trg_value[2+4*(t+m)] += tv2[m];
          trg_value[3+4*(t+m)] += tv3[m];
        }
      }
      for (; t<ntrgs; t++) {
        real_t potential = 0;
        vec3 gradient = 0;
        for (int s=0; s<nsrcs; s++) {
          vec3 dx;
          for (int d=0; d<3; d++)
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          real_t r2 = norm(dx);
          if (r2!=0) {
            real_t r = std::sqrt(r2);
            real_t kernel = std::exp(-wavek*r) / r * src_value[s];
            potential += kernel;
            real_t dpdr = - kernel * (wavek*r+1) / r;
            gradient[0] += dpdr / r * dx[0];
            gradient[1] += dpdr / r * dx[1];
            gradient[2] += dpdr / r * dx[2];
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
