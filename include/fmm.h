#ifndef fmm_h
#define fmm_h
#include <cstring>      // std::memset
#include <fstream>      // std::ofstream
#include <type_traits>  // std::is_same
#include "fmm_base.h"
#include "intrinsics.h"
#include "math_wrapper.h"

namespace exafmm_t {
  template <typename T>
  class Fmm : public FmmBase<T> {
  public:
    /* precomputation matrices */
    std::vector<std::vector<T>> matrix_UC2E_U;
    std::vector<std::vector<T>> matrix_UC2E_V;
    std::vector<std::vector<T>> matrix_DC2E_U;
    std::vector<std::vector<T>> matrix_DC2E_V;
    std::vector<std::vector<std::vector<T>>> matrix_M2M;
    std::vector<std::vector<std::vector<T>>> matrix_L2L;

    std::vector<M2LData> m2ldata;

    /* constructors */
    Fmm() {}
    Fmm(int p_, int ncrit_, int depth_) : FmmBase<T>(p_, ncrit_, depth_) {}

    /* precomputation */
    //! Setup the sizes of precomputation matrices
    void initialize_matrix() {
      int nsurf = this->nsurf;
      int depth = this->depth;
      matrix_UC2E_V.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_UC2E_U.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_DC2E_V.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_DC2E_U.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_M2M.resize(depth+1);
      matrix_L2L.resize(depth+1);
      for (int level=0; level<=depth; ++level) {
        matrix_M2M[level].resize(REL_COORD[M2M_Type].size(), std::vector<T>(nsurf*nsurf));
        matrix_L2L[level].resize(REL_COORD[L2L_Type].size(), std::vector<T>(nsurf*nsurf));
      }
    }

    //! Precompute M2M and L2L
    void precompute_M2M() {
      int nsurf = this->nsurf;
      real_t parent_coord[3] = {0, 0, 0};
      for (int level=0; level<=this->depth; level++) {
        RealVec parent_up_check_surf = surface(this->p, this->r0, level, parent_coord, 2.95);
        real_t s = this->r0 * powf(0.5, level+1);
        int npos = REL_COORD[M2M_Type].size();  // number of relative positions
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
          gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_U[level][0]), &matrix_pc2ce[0], &buffer[0]);
          gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &(matrix_M2M[level][i][0]));
          // L2L
          matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
          gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &(matrix_DC2E_V[level][0]), &buffer[0]);
          gemm(nsurf, nsurf, nsurf, &buffer[0], &(matrix_DC2E_U[level][0]), &(matrix_L2L[level][i][0]));
        }
      }
    }

    //! Precompute UC2UE and DC2DE matrices
    void precompute_check2equiv() {}

    //! Precompute M2L
    void precompute_M2L(std::ofstream& file) {}

    //! Save precomputation matrices
    void save_matrix(std::ofstream& file) {
      file.write(reinterpret_cast<char*>(&this->r0), sizeof(real_t));  // r0
      size_t size = this->nsurf * this->nsurf;
      for(int l=0; l<=this->depth; l++) {
        // UC2E, DC2E
        file.write(reinterpret_cast<char*>(&matrix_UC2E_U[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_UC2E_V[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_DC2E_U[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_DC2E_V[l][0]), size*sizeof(T));
        // M2M, L2L
        for (auto & vec : matrix_M2M[l]) {
          file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
        }
        for (auto & vec : matrix_L2L[l]) {
          file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
        }
      }
    }

    //! Check and load precomputation matrices
    void load_matrix() {
      int nsurf = this->nsurf;
      int depth = this->depth;
      size_t size_M2L = this->nfreq * 2 * NCHILD * NCHILD;
      size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * nsurf * nsurf * (depth+1) * sizeof(T) 
                       + REL_COORD[M2L_Type].size() * size_M2L * depth * sizeof(real_t)
                       + 1 * sizeof(real_t);   // +1 denotes r0
      std::ifstream file(this->filename, std::ifstream::binary);
      if (file.good()) {
        file.seekg(0, file.end);
        if (size_t(file.tellg()) == file_size) {   // if file size is correct
          file.seekg(0, file.beg);  // move the position back to the beginning
          real_t r0_;
          file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
          if (this->r0 == r0_) {    // if radius match
            size_t size = nsurf * nsurf;
            for (int l=0; l<depth; l++) {
              // UC2E, DC2E
              file.read(reinterpret_cast<char*>(&matrix_UC2E_U[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_UC2E_V[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_DC2E_U[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_DC2E_V[l][0]), size*sizeof(T));
              // M2M, L2L
              for (auto& vec : matrix_M2M[l]) {
                file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
              }
              for (auto& vec : matrix_L2L[l]) {
                file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
              }
            }
            this->is_precomputed = true;
          }
        }
      }
      file.close();
    }
    
    //! Precompute
    void precompute() {
      initialize_matrix();
      load_matrix();
      if (!this->is_precomputed) {
        precompute_check2equiv();
        precompute_M2M();
        std::remove(this->filename.c_str());
        std::ofstream file(this->filename, std::ofstream::binary);
        save_matrix(file);
        precompute_M2L(file);
        file.close();
      }
    }

    //! P2M operator
    void P2M(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> up_check_surf;
      up_check_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        up_check_surf[level].resize(nsurf*3);
        up_check_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // calculate upward check potential induced by sources' charges
        RealVec check_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
          check_coord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
          check_coord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
          check_coord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
        }
        this->potential_P2P(leaf->src_coord, leaf->src_value,
                            check_coord, leaf->up_equiv);
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_UC2E_U[level][0]), &(leaf->up_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->up_equiv[k] = equiv[k];
      }
    }

    //! L2P operator
    void L2P(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> dn_equiv_surf;
      dn_equiv_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        dn_equiv_surf[level].resize(nsurf*3);
        dn_equiv_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // down check surface potential -> equivalent surface charge
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_DC2E_U[level][0]), &(leaf->dn_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_DC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->dn_equiv[k] = equiv[k];
        // equivalent surface charge -> target potential
        RealVec equiv_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
          equiv_coord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->x[0];
          equiv_coord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->x[1];
          equiv_coord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->x[2];
        }
        this->gradient_P2P(equiv_coord, leaf->dn_equiv,
                           leaf->trg_coord, leaf->trg_value);
      }
    }

    //! M2M operator
    void M2M(Node<T>* node) {
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_M2M[level][octant][0]), &child->up_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++) {
            node->up_equiv[k] += buffer[k];
          }
        }
      }
    }
  
    //! L2L operator
    void L2L(Node<T>* node) {
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_L2L[level][octant][0]), &node->dn_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++)
            child->dn_equiv[k] += buffer[k];
        }
      }
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          L2L(node->children[octant]);
      }
#pragma omp taskwait
    }

    void M2L(Nodes<T>& nodes) {}
  };
  
  /** Below are member function specializations
   */
  template <>
  void Fmm<real_t>::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;
#pragma omp parallel for
    for (int level=0; level<=this->depth; ++level) {
      // compute kernel matrix
      RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
      RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
      RealVec matrix_c2e(nsurf*nsurf);  // UC2UE
      this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

      // svd
      RealVec S(nsurf*nsurf);  // singular values 
      RealVec U(nsurf*nsurf), VT(nsurf*nsurf);
      svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VT[0]);

      // pseudo-inverse
      real_t max_S = 0;
      for (int i=0; i<nsurf; i++) {
        max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
      }
      for (int i=0; i<nsurf; i++) {
        S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
      }
      RealVec V = transpose(VT, nsurf, nsurf);
      matrix_UC2E_U[level] = transpose(U, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &(matrix_UC2E_V[level][0]));
      matrix_DC2E_U[level] = VT;
      gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &(matrix_DC2E_V[level][0]));
    }
  }

  template <>
  void Fmm<complex_t>::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;
#pragma omp parallel for
    for (int level=0; level<=this->depth; ++level) {
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
      matrix_UC2E_U[level] = UH;
      gemm(nsurf, nsurf, nsurf, &V[0], &S_[0], &(matrix_UC2E_V[level][0]));
      matrix_DC2E_U[level] = transpose(V, nsurf, nsurf);
      ComplexVec UHT = transpose(UH, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &UHT[0], &S_[0], &(matrix_DC2E_V[level][0]));
    }
  } 

  //! member function specialization for real type
  template <>
  void Fmm<real_t>::precompute_M2L(std::ofstream& file) {
    int n1 = this->p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    int fft_size = 2 * nfreq * NCHILD * NCHILD;
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*nfreq));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fft plan
    RealVec fftw_in(nconv);
    RealVec fftw_out(2*nfreq);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft_r2c(3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()), FFTW_ESTIMATE);
    RealVec trg_coord(3,0);
    for (int l=1; l<this->depth+1; ++l) {
      // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
        real_t coord[3];
        for (int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 * powf(0.5, l-1);  // relative coords
        }
        RealVec conv_coord = convolution_grid(this->p, this->r0, l, coord);   // convolution grid
        RealVec conv_value(nconv);   // potentials on convolution grid
        this->kernel_matrix(conv_coord, trg_coord, conv_value);
        fft_execute_dft_r2c(plan, conv_value.data(), reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
      }
      // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
        for (int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
          int child_rel_idx = M2L_INDEX_MAP[i][j];
          if (child_rel_idx != -1) {
            for (int k=0; k<nfreq; k++) {   // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / nconv;   // real
              matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / nconv;   // imag
            }
          }
        }
      }
      // write to file
      for(auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()), fft_size*sizeof(real_t));
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  //! member function specialization for complex type
  template <>
  void Fmm<complex_t>::precompute_M2L(std::ofstream& file) {
    int n1 = this->p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    int fft_size = 2 * nfreq * NCHILD * NCHILD;
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*nfreq));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fft plan
    RealVec fftw_in(nconv);
    RealVec fftw_out(2*nfreq);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft(3, dim,
                                 reinterpret_cast<fft_complex*>(fftw_in.data()),
                                 reinterpret_cast<fft_complex*>(fftw_out.data()),
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    RealVec trg_coord(3,0);
    for (int l=1; l<this->depth+1; ++l) {
      // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
        real_t coord[3];
        for (int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 * powf(0.5, l-1);  // relative coords
        }
        RealVec conv_coord = convolution_grid(this->p, this->r0, l, coord);   // convolution grid
        ComplexVec conv_value(nconv);   // potentials on convolution grid
        this->kernel_matrix(conv_coord, trg_coord, conv_value);
        fft_execute_dft(plan, reinterpret_cast<fft_complex*>(conv_value.data()),
                              reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
      }
      // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
        for (int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
          int child_rel_idx = M2L_INDEX_MAP[i][j];
          if (child_rel_idx != -1) {
            for (int k=0; k<nfreq; k++) {   // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / nconv;   // real
              matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / nconv;   // imag
            }
          }
        }
      }
      // write to file
      for(auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()), fft_size*sizeof(real_t));
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }
}  // end namespace
#endif
