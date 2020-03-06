#ifndef fmm_scale_invariant
#define fmm_scale_invariant
#include <type_traits>  // std::is_same
#include "fmm_base.h"
#include "math_wrapper.h"

namespace exafmm_t {
  template <typename T>
  class FmmScaleInvariant : public FmmBase<T> {
    /** For the variables from base class that do not template parameter T,
     *  we need to use this-> to tell compilers to lookup nondependent names
     *  in the base class. Eg. p, nsurf, r0, kernel_matrix etc.
     *  https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members/
     */

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
      int n1 = this->p * 2;
      int n3_ = n1 * n1 * (n1 / 2 + 1);
      size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
      int nsurf = this->nsurf;
      matrix_UC2E_U.resize(nsurf*nsurf);
      matrix_UC2E_V.resize(nsurf*nsurf);
      matrix_DC2E_U.resize(nsurf*nsurf);
      matrix_DC2E_V.resize(nsurf*nsurf);
      matrix_M2M.resize(REL_COORD[M2M_Type].size(), std::vector<T>(nsurf*nsurf));
      matrix_L2L.resize(REL_COORD[L2L_Type].size(), std::vector<T>(nsurf*nsurf));
      matrix_M2L.resize(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));    
    }

    //! Precompute M2M and L2L
    void precompute_M2M() {
      int nsurf = this->nsurf;
      int npos = REL_COORD[M2M_Type].size();  // number of relative positions
      int level = 0;
      real_t parent_coord[3] = {0, 0, 0};
      RealVec parent_up_check_surf = surface(this->p, this->r0, level, parent_coord, 2.95);
      real_t s = this->r0 * powf(0.5, level+1);
#pragma omp parallel for
      for(int i=0; i<npos; i++) {
        // compute kernel matrix
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(this->p, this->r0, level+1, child_coord, 1.05);
        std::vector<T> matrix_pc2ce(nsurf*nsurf);
        this->kernel_matrix(parent_up_check_surf, child_up_equiv_surf, matrix_pc2ce);
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

    //! Precompute UC2UE and DC2DE matrices
    void precompute_check2equiv() {}

    //! Precompute M2L
    void precompute_M2L() {}

    //! Precompute
    void precompute() {
      initialize_matrix();
      precompute_check2equiv();
      precompute_M2M();
      precompute_M2L();
    }

    /* constructors */
    FmmScaleInvariant() {}
    FmmScaleInvariant(int p_, int ncrit_, int depth_) : FmmBase<T>(p_, ncrit_, depth_) {
    }

  };

  
  /** Below are member function specializations
   */
  template <>
  void FmmScaleInvariant<real_t>::precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;

    // compute kernel matrix
    RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
    RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
    RealVec matrix_c2e(nsurf*nsurf);  // UC2UE
    this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

    // svd
    RealVec S(nsurf*nsurf);  // singular values 
    RealVec U(nsurf*nsurf), VH(nsurf*nsurf);
    svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

    // pseudo-inverse
    real_t max_S = 0;
    for (int i=0; i<nsurf; i++) {
      max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
    }
    for (int i=0; i<nsurf; i++) {
      S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
    }
    RealVec V = transpose(VH, nsurf, nsurf);
    matrix_UC2E_U = transpose(U, nsurf, nsurf);
    gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &matrix_UC2E_V[0]);
    matrix_DC2E_U = VH;
    gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &matrix_DC2E_V[0]);
  }

  template <>
  void FmmScaleInvariant<complex_t>::precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;

    // compute kernel matrix
    RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
    RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
    ComplexVec matrix_c2e(nsurf*nsurf);  // UC2UE
    this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

    // svd
    RealVec S(nsurf*nsurf);  // singular values 
    ComplexVec U(nsurf*nsurf), VH(nsurf*nsurf);
    svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

    // pseudo-inverse
    real_t max_S = 0;
    for (int i=0; i<nsurf; i++) {
      max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
    }
    for (int i=0; i<nsurf; i++) {
      S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
    }
    ComplexVec S_(nsurf*nsurf);
    for (size_t i=0; i<S_.size(); i++) {   // convert S to complex type
      S_[i] = S[i];
    }
    ComplexVec V = conjugate_transpose(VH, nsurf, nsurf);
    ComplexVec UH = conjugate_transpose(U, nsurf, nsurf);
    matrix_UC2E_U = UH;
    gemm(nsurf, nsurf, nsurf, &V[0], &S_[0], &matrix_UC2E_V[0]);
    matrix_DC2E_U = transpose(V, nsurf, nsurf);
    ComplexVec UHT = transpose(UH, nsurf, nsurf);
    gemm(nsurf, nsurf, nsurf, &UHT[0], &S_[0], &matrix_DC2E_V[0]);
  }

  //! member function specialization for real type
  template <>
  void FmmScaleInvariant<real_t>::precompute_M2L() {
    int n1 = this->p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*n3_));
    // create fft plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft_r2c(3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()), FFTW_ESTIMATE);
    // compute M2L kernel matrix, perform DFT
    RealVec trg_coord(3,0);
#pragma omp parallel for
    for(size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 / 0.5;  // relative coords
      }
      RealVec conv_coord = convolution_grid(this->p, this->r0, 0, coord);   // convolution grid
      RealVec conv_value(n3);   // potentials on convolution grid
      this->kernel_matrix(conv_coord, trg_coord, conv_value);
      fft_execute_dft_r2c(plan, conv_value.data(), reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
    for(size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
      for(int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
        int child_rel_idx = M2L_INDEX_MAP[i][j];
        if (child_rel_idx != -1) {
          for(int k=0; k<n3_; k++) {   // loop over frequencies
            int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
            matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / n3;   // real
            matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / n3;   // imag
          }
        }
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

}  // end namespace
#endif
