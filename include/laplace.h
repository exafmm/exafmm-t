#ifndef laplace_h
#define laplace_h
#include "exafmm_t.h"
#include "fmm_scale_invariant.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

namespace exafmm_t {
  //! A derived FMM class for Laplace kernel.
  class LaplaceFmm : public FmmScaleInvariant<real_t> {
  public:
    LaplaceFmm() {}
    LaplaceFmm(int p_, int ncrit_, int depth_, std::string filename_=std::string()) :
      FmmScaleInvariant<real_t>(p_, ncrit_, depth_, filename_)
    {
      if (this->filename.empty()) {
        this->filename = std::string("laplace_") + (std::is_same<real_t, float>::value ? "f" : "d")
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
      real_t newton_coef = 16;   // comes from Newton's method in simd rsqrt function
      simdvec coef(real_t(1.0/(4*PI*newton_coef)));
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
          tv += invr * sv;
        }
        tv *= coef;
        for (int m=0; m<NSIMD && t+m<ntrgs; m++) {
          trg_value[t+m] += tv[m];
        }
      }
      for (; t<ntrgs; t++) {
        real_t potential = 0;
        for (int s=0; s<nsrcs; ++s) {
          vec3 dx = 0;
          for (int d=0; d<3; d++) {
            dx[d] += trg_coord[3*t+d] - src_coord[3*s+d];
          }
          real_t r2 = norm(dx);
          if (r2!=0) {
            real_t invr = 1 / std::sqrt(r2);
            potential += src_value[s] * invr;
          }
        }
        trg_value[t] += potential / (4*PI);
      }
      add_flop((long long)ntrgs*(long long)nsrcs*(12+4*2));
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
      real_t newton_coef = 16;   // comes from Newton's method in simd rsqrt function
      simdvec coefp(real_t(1.0/(4*PI*newton_coef)));
      simdvec coefg(real_t(1.0/(4*PI*newton_coef*newton_coef*newton_coef)));
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
          simdvec r2(zero);
          r2 += sx * sx;
          r2 += sy * sy;
          r2 += sz * sz;
          simdvec invr = rsqrt(r2);
          invr &= r2 > zero;
          simdvec invr3 = (invr*invr) * invr;
          simdvec sv(src_value[s]);
          tv0 += sv * invr;
          sv *= invr3;
          tv1 += sv * sx;
          tv2 += sv * sy;
          tv3 += sv * sz;
        }
        tv0 *= coefp;
        tv1 *= coefg;
        tv2 *= coefg;
        tv3 *= coefg;
        for (int m=0; m<NSIMD && t+m<ntrgs; m++) {
          trg_value[0+4*(t+m)] += tv0[m];
          trg_value[1+4*(t+m)] += tv1[m];
          trg_value[2+4*(t+m)] += tv2[m];
          trg_value[3+4*(t+m)] += tv3[m];
        }
      }
      for (; t<ntrgs; t++) {
        real_t potential = 0;
        vec3 gradient = 0;
        for (int s=0; s<nsrcs; ++s) {
          vec3 dx = 0;
          for (int d=0; d<3; ++d) {
            dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
          }
          real_t r2 = norm(dx);
          if (r2!=0) {
            real_t invr2 = 1.0 / r2;
            real_t invr = src_value[s] * std::sqrt(invr2);
            potential += invr;
            dx *= invr2 * invr;
            gradient[0] += dx[0];
            gradient[1] += dx[1];
            gradient[2] += dx[2];
          }
        }
        trg_value[4*t] += potential / (4*PI) ;
        trg_value[4*t+1] -= gradient[0] / (4*PI);
        trg_value[4*t+2] -= gradient[1] / (4*PI);
        trg_value[4*t+3] -= gradient[2] / (4*PI);
      }   
      add_flop((long long)ntrgs*(long long)nsrcs*(20+4*2));
    }
  };
}  // end namespace exafmm_t
#endif
