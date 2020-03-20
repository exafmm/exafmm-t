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
    ModifiedHelmholtzFmm(int p_, int ncrit_, int depth_, real_t wavek_) : Fmm(p_, ncrit_, depth_) {
      wavek = wavek_;
      this->filename = std::string("modified_helmholtz_") + (std::is_same<real_t, float>::value ? "f" : "d")
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
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      for (int i=0; i<ntrgs; ++i) {
        vec3 x_trg;
        real_t potential = 0;
        for (int d=0; d<3; ++d)
          x_trg[d] = trg_coord[3*i+d];
        for (int j=0; j<nsrcs; ++j) {
          vec3 x_src;
          for (int d=0; d<3; ++d) {
            x_src[d] = src_coord[3*j+d];
          }
          vec3 dx = x_trg - x_src;
          real_t r = std::sqrt(norm(dx));
          if (r>0) {
            potential += std::exp(-wavek*r) / r * src_value[j];
          }
        }
        trg_value[i] += potential / 4*PI;
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
      int nsrcs = src_coord.size() / 3;
      int ntrgs = trg_coord.size() / 3;
      for (int i=0; i<ntrgs; ++i) {
        vec3 x_trg;
        real_t potential = 0;
        vec3 gradient = 0;
        for (int d=0; d<3; ++d)
          x_trg[d] = trg_coord[3*i+d];
        for (int j=0; j<nsrcs; ++j) {
          vec3 x_src;
          for (int d=0; d<3; ++d) {
            x_src[d] = src_coord[3*j+d];
          }
          vec3 dx = x_trg - x_src;
          real_t r = std::sqrt(norm(dx));
          // dp / dr
          if (r>0) {
            real_t kernel  = std::exp(-wavek*r) / r;
            real_t dpdr = - kernel * (wavek*r+1) / r;
            potential += kernel * src_value[j];
            gradient[0] += dpdr / r * dx[0] * src_value[j];
            gradient[1] += dpdr / r * dx[1] * src_value[j];
            gradient[2] += dpdr / r * dx[2] * src_value[j];
          }
        }
        trg_value[4*i+0] += potential / 4*PI;
        trg_value[4*i+1] += gradient[0] / 4*PI;
        trg_value[4*i+2] += gradient[1] / 4*PI;
        trg_value[4*i+3] += gradient[2] / 4*PI;
      }
    }
  };
}  // end namespace exafmm_t
#endif
