#include "precompute_helmholtz.h"

namespace exafmm_t {
  std::vector<ComplexVec> matrix_UC2E_U, matrix_UC2E_V;
  std::vector<ComplexVec> matrix_DC2E_U, matrix_DC2E_V;
  std::vector<std::vector<ComplexVec>> matrix_M2M, matrix_L2L;
  std::vector<std::vector<RealVec>> matrix_M2L_Helper;
  std::vector<std::vector<AlignedVec>> matrix_M2L;
  std::string FILE_NAME;
  bool IS_PRECOMPUTED = true;  // whether matrices are precomputed

  // complex gemm by blas lib
  void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C) {
    char transA = 'N', transB = 'N';
    complex_t alpha(1., 0.), beta(0.,0.);
#if FLOAT
    cgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    zgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  void initialize_matrix() {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    size_t fft_size = n3 * 2 * NCHILD * NCHILD;
    matrix_UC2E_V.resize(MAXLEVEL+1);
    matrix_UC2E_U.resize(MAXLEVEL+1);
    matrix_DC2E_V.resize(MAXLEVEL+1);
    matrix_DC2E_U.resize(MAXLEVEL+1);
    matrix_M2M.resize(MAXLEVEL+1);
    matrix_L2L.resize(MAXLEVEL+1);
    matrix_M2L.resize(MAXLEVEL);  // skip the last level
    for(int level = 0; level <= MAXLEVEL; level++) {
      matrix_UC2E_V[level].resize(NSURF*NSURF);
      matrix_UC2E_U[level].resize(NSURF*NSURF);
      matrix_DC2E_V[level].resize(NSURF*NSURF);
      matrix_DC2E_U[level].resize(NSURF*NSURF);
      matrix_M2M[level].resize(REL_COORD[M2M_Type].size(), ComplexVec(NSURF*NSURF));
      matrix_L2L[level].resize(REL_COORD[L2L_Type].size(), ComplexVec(NSURF*NSURF));
    }
    for(int level = 0; level < MAXLEVEL; level++) {
      matrix_M2L[level].resize(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));  // N3 by (2*NCHILD*NCHILD) matrix
    }
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    ComplexVec WORK(LWORK);
    RealVec RWORK(5*k);
#if FLOAT
    cgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
#else
    zgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  ComplexVec transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  ComplexVec conjugate_transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = std::conj(vec[i*n+j]);
      }
    }
    return temp;
  }

  void precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    for(int level = 0; level <= MAXLEVEL; level++) {
      // caculate matrix_UC2E_U and matrix_UC2E_V
      RealVec up_check_surf = surface(P, c, 2.95, level);
      RealVec up_equiv_surf = surface(P, c, 1.05, level);
      ComplexVec matrix_c2e(NSURF*NSURF);
      kernel_matrix(&up_check_surf[0], NSURF, &up_equiv_surf[0], NSURF, &matrix_c2e[0]);
      RealVec S(NSURF*NSURF);
      ComplexVec S_(NSURF*NSURF);
      ComplexVec U(NSURF*NSURF), VH(NSURF*NSURF);
      svd(NSURF, NSURF, &matrix_c2e[0], &S[0], &U[0], &VH[0]);
      // inverse S
      real_t max_S = 0;
      for(int i=0; i<NSURF; i++) {
        max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
      }
      for(int i=0; i<NSURF; i++) {
        S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
      }
      for(size_t i=0; i<S.size(); ++i) {
        S_[i] = S[i];
      }
      // save matrix
      ComplexVec V = conjugate_transpose(VH, NSURF, NSURF);
      ComplexVec UH = conjugate_transpose(U, NSURF, NSURF);
      matrix_UC2E_U[level] = UH;
      gemm(NSURF, NSURF, NSURF, &V[0], &S_[0], &(matrix_UC2E_V[level][0]));

      matrix_DC2E_U[level] = transpose(V, NSURF, NSURF);
      ComplexVec UTH = transpose(UH, NSURF, NSURF);
      gemm(NSURF, NSURF, NSURF, &UTH[0], &S_[0], &(matrix_DC2E_V[level][0]));
    }
  }

  void precompute_M2M() {
    real_t parent_coord[3] = {0, 0, 0};
    for(int level = 0; level <= MAXLEVEL; level++) {
      RealVec parent_up_check_surf = surface(P, parent_coord, 2.95, level);
      real_t s = R0 * powf(0.5, level+1);
      int numRelCoord = REL_COORD[M2M_Type].size();
#pragma omp parallel for
      for(int i=0; i<numRelCoord; i++) {
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(P, child_coord, 1.05, level+1);
        ComplexVec matrix_pc2ce(NSURF*NSURF);
        kernel_matrix(&parent_up_check_surf[0], NSURF, &child_up_equiv_surf[0], NSURF, &matrix_pc2ce[0]);
        // M2M: child's upward_equiv to parent's check
        ComplexVec buffer(NSURF*NSURF);
        gemm(NSURF, NSURF, NSURF, &(matrix_UC2E_U[level][0]), &matrix_pc2ce[0], &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &(matrix_UC2E_V[level][0]), &buffer[0], &(matrix_M2M[level][i][0]));
        // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
        matrix_pc2ce = transpose(matrix_pc2ce, NSURF, NSURF);
        gemm(NSURF, NSURF, NSURF, &matrix_pc2ce[0], &(matrix_DC2E_V[level][0]), &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &buffer[0], &(matrix_DC2E_U[level][0]), &(matrix_L2L[level][i][0]));
      }
    }
  }

  void precompute_M2Lhelper() {
    matrix_M2L_Helper.resize(MAXLEVEL+1);
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    // create fftw plan
    ComplexVec fftw_in(n3);
    RealVec fftw_out(2*n3);
    int dim[3] = {2*P, 2*P, 2*P};
    fft_plan plan = fft_plan_dft(3, dim, reinterpret_cast<fft_complex*>(&fftw_in[0]), 
                                (fft_complex*)(&fftw_out[0]), FFTW_FORWARD, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = REL_COORD[M2L_Helper_Type].size();
    RealVec r_trg(3, 0.0);
    for(int l=0; l<=MAXLEVEL; l++) {
      matrix_M2L_Helper[l].resize(numRelCoord);
      #pragma omp parallel for
      for(int i=0; i<numRelCoord; i++) {
        real_t coord[3];
        for(int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * R0 * powf(0.5, l-1);
        }
        RealVec conv_coord = convolution_grid(coord, l);
        ComplexVec conv_poten(n3);
        kernel_matrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
        matrix_M2L_Helper[l][i].resize(2*n3);
        fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&conv_poten[0]), (fft_complex*)(&matrix_M2L_Helper[l][i][0]));
      }
    }
    fft_destroy_plan(plan);
  }

  void precompute_M2L() {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    int numParentRelCoord = REL_COORD[M2L_Type].size();
    // parent rel, child rel -> m2l_helper_idx
    std::vector<std::vector<int>> index_mapping(numParentRelCoord, std::vector<int>(NCHILD*NCHILD));
    for(int i=0; i<numParentRelCoord; ++i) {
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
          index_mapping[i][j] = child_rel_idx;
        }
      }
    }
    // copy from matrix_M2L_Helper to matrix_M2L
    for(int l=0; l<MAXLEVEL; ++l) {
      for(int i=0; i<numParentRelCoord; ++i) {
        for(int j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int child_rel_idx = index_mapping[i][j];
          if (child_rel_idx != -1) {
            for(int k=0; k<n3; k++) {                      // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[l][i][new_idx+0] = matrix_M2L_Helper[l+1][child_rel_idx][k*2+0] / n3;   // real
              matrix_M2L[l][i][new_idx+1] = matrix_M2L_Helper[l+1][child_rel_idx][k*2+1] / n3;   // imag
            }
          }
        }
      }
    }
  }

  bool load_matrix() {
    std::ifstream file(FILE_NAME, std::ifstream::binary);
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    size_t fft_size = n3 * 2 * NCHILD * NCHILD;
    size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * NSURF * NSURF * (MAXLEVEL+1) * sizeof(complex_t)
                     +  REL_COORD[M2L_Type].size() * fft_size * MAXLEVEL * sizeof(real_t) + 1;    // +1 denotes R0
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
        for(int level = 0; level <= MAXLEVEL; level++) {
          // UC2E, DC2E
          file.read(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(complex_t));
          file.read(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(complex_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(complex_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(complex_t));
          // M2M, L2L
          for(auto & vec : matrix_M2M[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(complex_t));
          }
          for(auto & vec : matrix_L2L[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(complex_t));
          }
        }
        // M2L
        for(int level = 0; level < MAXLEVEL; level++) {
          for(auto & vec : matrix_M2L[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), fft_size*sizeof(real_t));
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

  void save_matrix() {
    std::remove(FILE_NAME.c_str());
    std::ofstream file(FILE_NAME, std::ofstream::binary);
    // R0
    file.write(reinterpret_cast<char*>(&R0), sizeof(real_t));
    size_t size = NSURF*NSURF;
    for(int level = 0; level <= MAXLEVEL; level++) {
      // save UC2E, DC2E precompute data
      file.write(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(complex_t));
      file.write(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(complex_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(complex_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(complex_t));
      // M2M, M2L precompute data
      for(auto & vec : matrix_M2M[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(complex_t));
      }
      for(auto & vec : matrix_L2L[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(complex_t));
      }
    }
    // M2L precompute data
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    size = n3 * 2 * NCHILD * NCHILD;
    for(int level = 0; level < MAXLEVEL; level++) {
      for(auto & vec : matrix_M2L[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
    }
  }

  void precompute() {
    // if matrix binary file exists
    FILE_NAME = "helmholtz";
    FILE_NAME += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(P) + "_" + "l" + std::to_string(MAXLEVEL);
    FILE_NAME += ".dat";
    initialize_matrix();
    if (load_matrix()) {
      IS_PRECOMPUTED = false;
      return;
    } else {
      precompute_check2equiv();
      precompute_M2M();
      precompute_M2Lhelper();
      precompute_M2L();
      save_matrix();
    }
  }
}//end namespace
