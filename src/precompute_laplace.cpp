#include "precompute_laplace.h"

namespace exafmm_t {
  RealVec matrix_UC2E_U, matrix_UC2E_V;
  RealVec matrix_DC2E_U, matrix_DC2E_V;
  std::vector<RealVec> matrix_M2M, matrix_L2L;
  std::vector<AlignedVec> matrix_M2L;
  std::string FILE_NAME;
  bool IS_PRECOMPUTED = true;  // whether matrices are precomputed

  // Map indices of M2L_Type to indices of M2L_Helper_Type
  std::vector<std::vector<int>> map_matrix_index() {
    int num_coords = REL_COORD[M2L_Type].size();   // number of relative coords for M2L_Type
    std::vector<std::vector<int>> parent2child(num_coords, std::vector<int>(NCHILD*NCHILD));
#pragma omp parallel for
    for(int i=0; i<num_coords; ++i) {
      for(int j1=0; j1<NCHILD; ++j1) {
        for(int j2=0; j2<NCHILD; ++j2) {
          ivec3& parent_rel_coord = REL_COORD[M2L_Type][i];
          ivec3  child_rel_coord;
          child_rel_coord[0] = parent_rel_coord[0]*2 - (j1/1)%2 + (j2/1)%2;
          child_rel_coord[1] = parent_rel_coord[1]*2 - (j1/2)%2 + (j2/2)%2;
          child_rel_coord[2] = parent_rel_coord[2]*2 - (j1/4)%2 + (j2/4)%2;
          int coord_hash = hash(child_rel_coord);
          int child_rel_idx = HASH_LUT[M2L_Helper_Type][coord_hash];
          int j = j2*NCHILD + j1;
          parent2child[i][j] = child_rel_idx;
        }
      }
    }
    return parent2child;
  }

  //! blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
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

  RealVec transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void initialize_matrix() {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    matrix_UC2E_U.resize(NSURF*NSURF);
    matrix_UC2E_V.resize(NSURF*NSURF);
    matrix_DC2E_U.resize(NSURF*NSURF);
    matrix_DC2E_V.resize(NSURF*NSURF);
    matrix_M2M.resize(REL_COORD[M2M_Type].size(), RealVec(NSURF*NSURF));
    matrix_L2L.resize(REL_COORD[L2L_Type].size(), RealVec(NSURF*NSURF));
    matrix_M2L.resize(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
  }

  void precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    // caculate upwardcheck to equiv U and V
    RealVec up_check_surf = surface(P, c, 2.95, level);
    RealVec up_equiv_surf = surface(P, c, 1.05, level);
    RealVec matrix_c2e(NSURF*NSURF);
    kernel_matrix(&up_check_surf[0], NSURF, &up_equiv_surf[0], NSURF, &matrix_c2e[0]);
    RealVec U(NSURF*NSURF), S(NSURF*NSURF), VT(NSURF*NSURF);
    svd(NSURF, NSURF, &matrix_c2e[0], &S[0], &U[0], &VT[0]);
    // inverse S
    real_t max_S = 0;
    for(int i=0; i<NSURF; i++) {
      max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
    }
    for(int i=0; i<NSURF; i++) {
      S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
    }
    // save matrix
    RealVec V = transpose(VT, NSURF, NSURF);
    matrix_UC2E_U = transpose(U, NSURF, NSURF);
    gemm(NSURF, NSURF, NSURF, &V[0], &S[0], &matrix_UC2E_V[0]);

    matrix_DC2E_U = VT;
    gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &matrix_DC2E_V[0]);
  }

  void precompute_M2M() {
    int numRelCoord = REL_COORD[M2M_Type].size();
    int level = 0;
    real_t parent_coord[3] = {0, 0, 0};
    RealVec parent_up_check_surf = surface(P, parent_coord, 2.95, level);
    real_t s = R0 * powf(0.5, level+1);
#pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      ivec3& coord = REL_COORD[M2M_Type][i];
      real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                               parent_coord[1] + coord[1]*s,
                               parent_coord[2] + coord[2]*s};
      RealVec child_up_equiv_surf = surface(P, child_coord, 1.05, level+1);
      RealVec matrix_pc2ce(NSURF*NSURF);
      kernel_matrix(&parent_up_check_surf[0], NSURF, &child_up_equiv_surf[0], NSURF, &matrix_pc2ce[0]);
      // M2M
      RealVec buffer(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &matrix_UC2E_U[0], &matrix_pc2ce[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &matrix_UC2E_V[0], &buffer[0], &(matrix_M2M[i][0]));
      // L2L
      matrix_pc2ce = transpose(matrix_pc2ce, NSURF, NSURF);
      gemm(NSURF, NSURF, NSURF, &matrix_pc2ce[0], &matrix_DC2E_V[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &buffer[0], &matrix_DC2E_U[0], &(matrix_L2L[i][0]));
    }
  }

  void precompute_M2L(std::vector<std::vector<int>>& parent2child) {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);

    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*n3_));
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*P, 2*P, 2*P};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, fftw_in.data(), nullptr, 1, n3,
                    reinterpret_cast<fft_complex*>(fftw_out.data()), nullptr, 1, n3_,
                    FFTW_ESTIMATE);
    // Precompute M2L matrix
    RealVec trg_coord(3,0);
    // compute DFT of potentials at convolution grids
#pragma omp parallel for
    for(int i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * R0 / 0.5;
      }
      RealVec conv_coord = convolution_grid(coord, 0);   // convolution grid
      RealVec conv_p(n3);   // potentials on convolution grid
      kernel_matrix(conv_coord.data(), n3, trg_coord.data(), 1, conv_p.data());
      fft_execute_dft_r2c(plan, conv_p.data(), 
          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L, reorder to improve data locality
#pragma omp parallel for
    for(int i=0; i<REL_COORD[M2L_Type].size(); ++i) {
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
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  bool load_matrix() {
    std::ifstream file(FILE_NAME, std::ifstream::binary);
    int n1 = P * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * NSURF * NSURF
                     +  REL_COORD[M2L_Type].size() * fft_size + 1;   // +1 denotes R0
    file_size *= sizeof(real_t);
    if (file.good()) {     // if file exists
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == file_size) {   // if file size is correct
        file.seekg(0, file.beg);         // move the position back to the beginning
        // check whether R0 matches
        real_t R0_;
        file.read(reinterpret_cast<char*>(&R0_), sizeof(real_t));
        if (R0 != R0_) {
          return false;
        }
        size_t size = NSURF * NSURF;
        // UC2E, DC2E
        file.read(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(real_t));
        // M2M, L2L
        for(auto & vec : matrix_M2M) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
        }
        for(auto & vec : matrix_L2L) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
        }
        // M2L
        size = n3_ * 2 * NCHILD * NCHILD;
        for(auto & vec : matrix_M2L) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
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

  void save_matrix() {
    std::remove(FILE_NAME.c_str());
    std::ofstream file(FILE_NAME, std::ofstream::binary);
    // R0
    file.write(reinterpret_cast<char*>(&R0), sizeof(real_t));
    size_t size = NSURF*NSURF;
    // UC2E, DC2E
    file.write(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(real_t));
    // M2M, L2L
    for(auto & vec : matrix_M2M) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    for(auto & vec : matrix_L2L) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    // M2L
    int n1 = P * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size = n3_ * 2 * NCHILD * NCHILD;
    for(auto & vec : matrix_M2L) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    file.close();
  }

  void precompute() {
    // if matrix binary file exists
    FILE_NAME = "laplace";
    FILE_NAME += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(P);
    FILE_NAME += ".dat";
    initialize_matrix();
    if (load_matrix()) {
      IS_PRECOMPUTED = false;
      return;
    } else {
      precompute_check2equiv();
      precompute_M2M();
      auto parent2child = map_matrix_index();
      precompute_M2L(parent2child);
      save_matrix();
    }
  }
}//end namespace
