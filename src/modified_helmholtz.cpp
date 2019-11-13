#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "modified_helmholtz.h"

namespace exafmm_t {
  //! blas gemv with row major data
  void ModifiedHelmholtzFMM::gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
    char trans = 'T';
    real_t alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
#if FLOAT
    sgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    dgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }

  //! blas gemm with row major data
  void ModifiedHelmholtzFMM::gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void ModifiedHelmholtzFMM::svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    RealVec WORK(LWORK);
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  RealVec ModifiedHelmholtzFMM::transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void ModifiedHelmholtzFMM::potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    for (int i=0; i<ntrgs; ++i) {
      vec3 x_trg;
      for (int d=0; d<3; ++d)
        x_trg[d] = trg_coord[3*i+d];
      for (int j=0; j<nsrcs; ++j) {
        vec3 x_src;
        for (int d=0; d<3; ++d)
          x_src[d] = src_coord[3*j+d];
        vec3 dx = x_trg - x_src;
        real_t r = std::sqrt(norm(dx));
        if (r>0)
          trg_value[i] += std::exp(-wavek*r) / r;
      }
      trg_value[i] /= 4*PI;
    }
  }

  void ModifiedHelmholtzFMM::gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    for (int i=0; i<ntrgs; ++i) {
      vec3 x_trg;
      for (int d=0; d<3; ++d)
        x_trg[d] = trg_coord[3*i+d];
      for (int j=0; j<nsrcs; ++j) {
        vec3 x_src;
        for (int d=0; d<3; ++d)
          x_src[d] = src_coord[3*j+d];
        vec3 dx = x_trg - x_src;
        real_t r = std::sqrt(norm(dx));
        // dp / dr
        if (r>0) {
          real_t dpdr = - std::exp(-wavek*r) * (wavek*r+1) / (r*r);
          trg_value[4*i+0] += std::exp(-wavek*r) / r;
          trg_value[4*i+1] += dpdr / r * dx[0];
          trg_value[4*i+2] += dpdr / r * dx[1];
          trg_value[4*i+3] += dpdr / r * dx[2];
        }
      }
      for (int d=0; d<4; ++d)
        trg_value[4*i+d] /= 4*PI;
    }
  }

  //! P2P save pairwise contributions to k_out (not aggregate over each target)
  void ModifiedHelmholtzFMM::kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1.);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0.);
      potential_P2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }

  void ModifiedHelmholtzFMM::initialize_matrix() {
    matrix_UC2E_V.resize(depth+1);
    matrix_UC2E_U.resize(depth+1);
    matrix_DC2E_V.resize(depth+1);
    matrix_DC2E_U.resize(depth+1);
    matrix_M2M.resize(depth+1);
    matrix_L2L.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      matrix_UC2E_V[level].resize(nsurf*nsurf);
      matrix_UC2E_U[level].resize(nsurf*nsurf);
      matrix_DC2E_V[level].resize(nsurf*nsurf);
      matrix_DC2E_U[level].resize(nsurf*nsurf);
      matrix_M2M[level].resize(REL_COORD[M2M_Type].size(), RealVec(nsurf*nsurf));
      matrix_L2L[level].resize(REL_COORD[L2L_Type].size(), RealVec(nsurf*nsurf));
    }
  }

  void ModifiedHelmholtzFMM::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    for(int level = 0; level <= depth; level++) {
      // caculate matrix_UC2E_U and matrix_UC2E_V
      RealVec up_check_surf = surface(p, r0, level, c, 2.95);
      RealVec up_equiv_surf = surface(p, r0, level, c, 1.05);
      RealVec matrix_c2e(nsurf*nsurf);
      kernel_matrix(&up_check_surf[0], nsurf, &up_equiv_surf[0], nsurf, &matrix_c2e[0]);
      RealVec U(nsurf*nsurf), S(nsurf*nsurf), VT(nsurf*nsurf);
      svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VT[0]);
      // inverse S
      real_t max_S = 0;
      for(int i=0; i<nsurf; i++) {
        max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
      }
      for(int i=0; i<nsurf; i++) {
        S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
      }
      // save matrix
      RealVec V = transpose(VT, nsurf, nsurf);
      matrix_UC2E_U[level] = transpose(U, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &(matrix_UC2E_V[level][0]));

      matrix_DC2E_U[level] = VT;
      gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &(matrix_DC2E_V[level][0]));
    }
  }

  void ModifiedHelmholtzFMM::precompute_M2M() {
    real_t parent_coord[3] = {0, 0, 0};
    for(int level = 0; level <= depth; level++) {
      RealVec parent_up_check_surf = surface(p, r0, level, parent_coord, 2.95);
      real_t s = r0 * powf(0.5, level+1);
      int num_coords = REL_COORD[M2M_Type].size();
#pragma omp parallel for
      for(int i=0; i<num_coords; i++) {
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(p, r0, level+1, child_coord, 1.05);
        RealVec matrix_pc2ce(nsurf*nsurf);
        kernel_matrix(&parent_up_check_surf[0], nsurf, &child_up_equiv_surf[0], nsurf, &matrix_pc2ce[0]);
        // M2M: child's upward_equiv to parent's check
        RealVec buffer(nsurf*nsurf);
        gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_U[level][0]), &matrix_pc2ce[0], &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &(matrix_M2M[level][i][0]));
        // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
        matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
        gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &(matrix_DC2E_V[level][0]), &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &buffer[0], &(matrix_DC2E_U[level][0]), &(matrix_L2L[level][i][0]));
      }
    }
  }

  void ModifiedHelmholtzFMM::precompute_M2L(std::ofstream& file, std::vector<std::vector<int>>& parent2child) {
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    int fft_size = 2 * n3_ * NCHILD * NCHILD;

    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(), RealVec(2*n3_));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*p, 2*p, 2*p};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, fftw_in.data(), nullptr, 1, n3, 
                    reinterpret_cast<fft_complex*>(fftw_out.data()), nullptr, 1, n3_,
                    FFTW_ESTIMATE);
    // Precompute M2L matrix
    RealVec trg_coord(3,0);
    for(int l=1; l<depth+1; ++l) {
      // compute DFT of potentials at convolution grids
#pragma omp parallel for
      for(size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); i++) {
        real_t coord[3];
        for(int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * r0 * powf(0.5, l-1);
        }
        RealVec conv_coord = convolution_grid(p, r0, l, coord);   // convolution grid
        RealVec conv_p(n3);   // potentials on convolution grid
        kernel_matrix(conv_coord.data(), n3, trg_coord.data(), 1, conv_p.data());
        fft_execute_dft_r2c(plan, conv_p.data(),
                            reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
      }
      // convert M2L_Helper to M2L, reorder to improve data locality
#pragma omp parallel for
      for(size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
        for(int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
          int child_rel_idx = parent2child[i][j];
          if (child_rel_idx != -1) {
            for(int k=0; k<n3_; k++) {   // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / n3;   // real
              matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / n3;   // imag
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

  bool ModifiedHelmholtzFMM::load_matrix() {
    std::ifstream file(filename, std::ifstream::binary);
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * nsurf * nsurf * (depth+1) * sizeof(real_t)
                     +  REL_COORD[M2L_Type].size() * fft_size * depth * sizeof(real_t) + 2 * sizeof(real_t);    // 2 denotes r0 and wavek
    if (file.good()) {     // if file exists
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == file_size) {   // if file size is correct
        file.seekg(0, file.beg);         // move the position back to the beginning
        // check whether r0 matches
        real_t r0_;
        file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
        if (r0 != r0_) {
          std::cout << r0 << " " << r0_ << std::endl;
          return false;
        }
        // check whether wavek matches
        real_t wavek_;
        file.read(reinterpret_cast<char*>(&wavek_), sizeof(real_t));
        if (wavek != wavek_) {
          std::cout << wavek << " " << wavek_ << std::endl;
          return false;
        }
        size_t size = nsurf * nsurf;
        for(int level = 0; level <= depth; level++) {
          // UC2E, DC2E
          file.read(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(real_t));
          // M2M, L2L
          for(auto & vec : matrix_M2M[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
          }
          for(auto & vec : matrix_L2L[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
          }
        }
        file.close();
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  void ModifiedHelmholtzFMM::save_matrix(std::ofstream& file) {
    // root radius r0
    file.write(reinterpret_cast<char*>(&r0), sizeof(real_t));
    // wavenumber wavek
    file.write(reinterpret_cast<char*>(&wavek), sizeof(real_t));
    size_t size = nsurf*nsurf;
    for(int level = 0; level <= depth; level++) {
      // save UC2E, DC2E precompute data
      file.write(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(real_t));
      // M2M, M2L precompute data
      for(auto & vec : matrix_M2M[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
      for(auto & vec : matrix_L2L[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
    }
  }

  void ModifiedHelmholtzFMM::precompute() {
    // if matrix binary file exists
    filename = "modified_helmholtz";
    filename += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(p) + "_" + "l" + std::to_string(depth);
    filename += ".dat";
    initialize_matrix();
    if (load_matrix()) {
      is_precomputed = false;
      return;
    } else {
      precompute_check2equiv();
      precompute_M2M();
      std::remove(filename.c_str());
      std::ofstream file(filename, std::ofstream::binary);
      save_matrix(file);
      auto parent2child = map_matrix_index();
      precompute_M2L(file, parent2child);
      file.close();
    }
  }
}  // end namespace exafmm_t
