#ifndef precompute_h
#define precompute_h
#include "exafmm_t.h"
#include "geometry.h"
#include "laplace.h"

namespace exafmm_t {
  RealVec UC2E_U, UC2E_V;
  RealVec DC2E_U, DC2E_V;
  std::vector<RealVec> matrix_M2L_Helper;
  std::vector<RealVec> matrix_M2M;
  std::vector<RealVec> matrix_M2L;
  std::vector<RealVec> matrix_L2L;

  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);
  RealVec transpose(RealVec& vec, int m, int n);

  void precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    // caculate upwardcheck to equiv U and V
    RealVec up_check_surf = surface(MULTIPOLE_ORDER,c,2.95,level);
    RealVec up_equiv_surf = surface(MULTIPOLE_ORDER,c,1.05,level);
    RealVec matrix_c2e(NSURF*NSURF);
    kernel_matrix(&up_check_surf[0], NSURF, &up_equiv_surf[0], NSURF, &matrix_c2e[0]);
    RealVec U(NSURF*NSURF), S(NSURF*NSURF), VT(NSURF*NSURF);
    svd(NSURF, NSURF, &matrix_c2e[0], &S[0], &U[0], &VT[0]);
    // inverse S
    real_t max_S = 0;
    for(size_t i=0; i<NSURF; i++) {
      max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
    }
    for(size_t i=0; i<NSURF; i++) {
      S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
    }
    // save matrix
    RealVec V = transpose(VT, NSURF, NSURF);
    UC2E_V.resize(NSURF*NSURF);
    UC2E_U = transpose(U, NSURF, NSURF);
    gemm(NSURF, NSURF, NSURF, &V[0], &S[0], &UC2E_V[0]);

    DC2E_V.resize(NSURF*NSURF);
    DC2E_U = VT;
    gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &DC2E_V[0]);
  }

  void precompute_M2M() {
    int numRelCoord = rel_coord[M2M_Type].size();
    matrix_M2M.resize(numRelCoord);
    matrix_L2L.resize(numRelCoord);
    int level = 0;
    real_t parent_coord[3] = {0, 0, 0};
    RealVec parent_up_check_surf = surface(MULTIPOLE_ORDER,parent_coord,2.95,level);
    real_t s = R0 * powf(0.5, level+1);
#pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      ivec3& coord = rel_coord[M2M_Type][i];
      real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
      RealVec child_up_equiv_surf = surface(MULTIPOLE_ORDER,child_coord,1.05,level+1);
      RealVec matrix_pc2ce(NSURF*NSURF);
      kernel_matrix(&parent_up_check_surf[0], NSURF, &child_up_equiv_surf[0], NSURF, &matrix_pc2ce[0]);
      // M2M
      RealVec buffer(NSURF*NSURF);
      matrix_M2M[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &UC2E_U[0], &matrix_pc2ce[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &UC2E_V[0], &buffer[0], &(matrix_M2M[i][0]));
      // L2L
      matrix_pc2ce = transpose(matrix_pc2ce, NSURF, NSURF);
      matrix_L2L[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &matrix_pc2ce[0], &DC2E_V[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &buffer[0], &DC2E_U[0], &(matrix_L2L[i][0]));
    }
  }

  void precompute_M2Lhelper() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, &fftw_in[0], nullptr, 1, n3,
                    (fft_complex*)(&fftw_out[0]), nullptr, 1, n3_, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = rel_coord[M2L_Helper_Type].size();
    matrix_M2L_Helper.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = rel_coord[M2L_Helper_Type][i][d] * R0 / 0.5;
      }
      RealVec conv_coord = convolution_grid(coord, 0);
      RealVec r_trg(3, 0.0);
      RealVec conv_poten(n3);
      kernel_matrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
      matrix_M2L_Helper[i].resize(2*n3_);
      fft_execute_dft_r2c(plan, &conv_poten[0], (fft_complex*)(&matrix_M2L_Helper[i][0]));
    }
    fft_destroy_plan(plan);
  }

  void precompute_M2L() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    int numParentRelCoord = rel_coord[M2L_Type].size();
    int numChildRelCoord = rel_coord[M2L_Helper_Type].size();
    matrix_M2L.resize(numParentRelCoord);
    RealVec zero_vec(n3_*2, 0);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<numParentRelCoord; i++) {
      ivec3& parentRelCoord = rel_coord[M2L_Type][i];
      std::vector<real_t*> matrix_ptr(NCHILD*NCHILD, &zero_vec[0]);
      for(int j1=0; j1<NCHILD; j1++) {
        for(int j2=0; j2<NCHILD; j2++) {
          int childRelCoord[3]= { parentRelCoord[0]*2 - (j1/1)%2 + (j2/1)%2,
                                  parentRelCoord[1]*2 - (j1/2)%2 + (j2/2)%2,
                                  parentRelCoord[2]*2 - (j1/4)%2 + (j2/4)%2 };
          for(int k=0; k<numChildRelCoord; k++) {
            ivec3& childRefCoord = rel_coord[M2L_Helper_Type][k];
            if (childRelCoord[0] == childRefCoord[0] &&
                childRelCoord[1] == childRefCoord[1] &&
                childRelCoord[2] == childRefCoord[2]) {
              matrix_ptr[j2*NCHILD+j1] = &matrix_M2L_Helper[k][0];
              break;
            }
          }
        }
      }
      matrix_M2L[i].resize(n3_*2*NCHILD*NCHILD);  // N3 by (2*NCHILD*NCHILD) matrix
      for(int k=0; k<n3_; k++) {                      // loop over frequencies
        for(size_t j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int index = k*(2*NCHILD*NCHILD) + 2*j;
          matrix_M2L[i][index+0] = matrix_ptr[j][k*2+0]/n3;   // real
          matrix_M2L[i][index+1] = matrix_ptr[j][k*2+1]/n3;   // imag
        }
      }
    }
  }

  void precompute() {
    precompute_check2equiv();
    precompute_M2M();
    precompute_M2Lhelper();
    precompute_M2L();
  }
}//end namespace
#endif
