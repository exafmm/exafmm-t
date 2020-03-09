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
    LaplaceFmm(int p_, int ncrit_, int depth_) : FmmScaleInvariant<real_t>(p_, ncrit_, depth_) {
      this->filename = std::string("laplace_") + (std::is_same<real_t, float>::value ? "f" : "d")
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
    void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
      simdvec zero(real_t(0));
      real_t newton_coef = 16;   // comes from Newton's method in simd rsqrt function
      const real_t COEF = 1.0/(4*PI*newton_coef);
      simdvec coef(COEF);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      for(t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv(zero);
        for(int s=0; s<nsrcs; s++) {
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
        for(int k=0; k<NSIMD && t+k<ntrgs; k++) {
          trg_value[t+k] += tv[k];
        }
      }
      for(; t<ntrgs; t++) {
        real_t potential = 0;
        for(int s=0; s<nsrcs; ++s) {
          vec3 dx = 0;
          for(int d=0; d<3; d++) {
            dx[d] += trg_coord[3*t+d] - src_coord[3*s+d];
          }
          real_t r2 = norm(dx);
          if(r2!=0) {
            real_t inv_r = 1 / std::sqrt(r2);
            potential += src_value[s] * inv_r;
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
      real_t newton_coefp = 16;   // comes from Newton's method in simd rsqrt function
      real_t newton_coefg = 16*16*16;
      const real_t COEFP = 1.0/(4*PI*newton_coefp);
      const real_t COEFG = -1.0/(4*PI*newton_coefg);
      simdvec coefp(COEFP);
      simdvec coefg(COEFG);
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      int t;
      for(t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
        simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
        simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
        simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
        simdvec tv0(zero);
        simdvec tv1(zero);
        simdvec tv2(zero);
        simdvec tv3(zero);
        for(int s=0; s<nsrcs; s++) {
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
        for(int k=0; k<NSIMD && t+k<ntrgs; k++) {
          trg_value[0+4*(t+k)] += tv0[k];
          trg_value[1+4*(t+k)] += tv1[k];
          trg_value[2+4*(t+k)] += tv2[k];
          trg_value[3+4*(t+k)] += tv3[k];
        }
      }
      for(; t<ntrgs; t++) {
        real_t potential = 0;
        vec3 gradient = 0;
        for(int s=0; s<nsrcs; ++s) {
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
        trg_value[4*t] += potential / (4*PI) ;
        trg_value[4*t+1] -= gradient[0] / (4*PI);
        trg_value[4*t+2] -= gradient[1] / (4*PI);
        trg_value[4*t+3] -= gradient[2] / (4*PI);
      }   
    }

    //! M2L operator.
    void M2L(Nodes<real_t>& nodes) {}

/*

    void M2L_setup(NodePtrs_t& nonleafs);

    void hadamard_product(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                          AlignedVec& fft_in, AlignedVec& fft_out);

    void fft_up_equiv(std::vector<size_t>& fft_offset, RealVec& all_up_equiv, AlignedVec& fft_in);

    void ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal, AlignedVec& fft_out, RealVec& all_dn_equiv);
    
*/

    /**
     * @brief Calculate the relative error of potentials and gradients in L2-norm.
     * 
     * @param leafs Vector of pointers to leaf nodes.
     * @return RealVec A two-element vector: potential error and gradient error.
     */
//    RealVec verify(NodePtrs_t& leafs);
  };
}  // end namespace exafmm_t
#endif
