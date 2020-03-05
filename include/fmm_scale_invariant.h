#ifndef fmm_scale_invariant
#define fmm_scale_invariant
#include <type_traits>  // std::is_same
#include "fmm_base.h"

namespace exafmm_t {
  template <typename T>
  class FmmScaleInvariant : public FmmBase<T> {
    using FmmBase<T>::p;
    using FmmBase<T>::nsurf;
    using FmmBase<T>::r0;
    using FmmBase<T>::is_real;

  public:
    /* precomputation matrices */
    std::vector<T> matrix_UC2E_U;  //!< First component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    std::vector<T> matrix_UC2E_V;  //!< Second component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    std::vector<T> matrix_DC2E_U;  //!< First component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    std::vector<T> matrix_DC2E_V;  //!< Second component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    std::vector<std::vector<T>> matrix_M2M;     //!< The pseudo-inverse of M2M kernel matrix.
    std::vector<std::vector<T>> matrix_L2L;     //!< The pseudo-inverse of L2L kernel matrix.
    std::vector<AlignedVec> matrix_M2L;  //!< The pseudo-inverse of M2L kernel matrix.

    M2LData m2ldata;

    /* precomputation */
    //! Setup the sizes of precomputation matrices
    void initialize_matrix() {
      int n1 = p * 2;
      int n3_ = n1 * n1 * (n1 / 2 + 1);
      size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
      matrix_UC2E_U.resize(nsurf*nsurf);
      matrix_UC2E_V.resize(nsurf*nsurf);
      matrix_DC2E_U.resize(nsurf*nsurf);
      matrix_DC2E_V.resize(nsurf*nsurf);
      matrix_M2M.resize(REL_COORD[M2M_Type].size(), std::vector<T>(nsurf*nsurf));
      matrix_L2L.resize(REL_COORD[L2L_Type].size(), std::vector<T>(nsurf*nsurf));
      matrix_M2L.resize(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));    
    }

    //! Precompute UC2UE and DC2DE matrices
    void precompute_check2equiv() {
      int level = 0;
      real_t c[3] = {0, 0, 0};

      // compute kernel matrix
      RealVec up_check_surf = surface(p, r0, level, c, 2.95);
      RealVec up_equiv_surf = surface(p, r0, level, c, 1.05);
      std::vector<T> matrix_c2e(nsurf*nsurf);  // UC2UE
      kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

      // svd
      RealVec S(nsurf*nsurf);  // singular values 
      std::vector<T> U(nsurf*nsurf), VH(nsurf*nsurf);
      svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

      // pseudo-inverse
      real_t max_S = 0;
      for (int i=0; i<nsurf; i++) {
        max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
      }
      for (int i=0; i<nsurf; i++) {
        S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
      }
      if (is_real) {  // T is real_t
        std::vector<T> V = transpose(VH, nsurf, nsurf);
        matrix_UC2E_U = transpose(U, nsurf, nsurf);
        gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &matrix_UC2E_V[0]);
        matrix_DC2E_U = VH;
        gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &matrix_DC2E_V[0]);
      } else {        // T is complex_t
        std::vector<T> S_(nsurf*nsurf);
        for (size_t i=0; i<S_.size(); i++) {   // convert S to complex type
          S_[i] = S[i];
        }
        std::vector<T> V = conjugate_transpose(VH, nsurf, nsurf);
        std::vector<T> UH = conjugate_transpose(U, nsurf, nsurf);
        matrix_UC2E_U = UH;
        gemm(nsurf, nsurf, nsurf, &V[0], &S_[0], &matrix_UC2E_V[0]);
        matrix_DC2E_U = transpose(V, nsurf, nsurf);
        std::vector<T> UHT = transpose(UH, nsurf, nsurf);
        gemm(nsurf, nsurf, nsurf, &UHT[0], &S_[0], &matrix_DC2E_V[0]);
      }
    }

    //! Precompute M2M and L2L
    void precompute_M2M() {
      int npos = REL_COORD[M2M_Type].size();  // number of relative positions
      int level = 0;
      real_t parent_coord[3] = {0, 0, 0};
      RealVec parent_up_check_surf = surface(p, r0, level, parent_coord, 2.95);
      real_t s = r0 * powf(0.5, level+1);
#pragma omp parallel for
      for(int i=0; i<npos; i++) {
        // compute kernel matrix
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(p, r0, level+1, child_coord, 1.05);
        std::vector<T> matrix_pc2ce(nsurf*nsurf);
        kernel_matrix(parent_up_check_surf, child_up_equiv_surf, matrix_pc2ce);
        // M2M
        std::vector<T> buffer(nsurf*nsurf);
        gemm(nsurf, nsurf, nsurf, &matrix_UC2E_U[0], &matrix_pc2ce[0], &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &matrix_UC2E_V[0], &buffer[0], &(matrix_M2M[i][0]));
        // L2L
        matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
        gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &matrix_DC2E_V[0], &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &buffer[0], &matrix_DC2E_U[0], &(matrix_L2L[i][0]));
      }
    }

    /* constructors */
    FmmScaleInvariant() {}
    FmmScaleInvariant(int p_, int ncrit_, int depth_) : FmmBase<T>(p_, ncrit_, depth_) {
    }


  };

}  // end namespace

#endif
