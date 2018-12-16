#ifndef precompute_h
#define precompute_h
#include "exafmm_t.h"
#include "geometry.h"
#include "laplace.h"

namespace exafmm_t {
  RealVec M2M_U, M2M_V;
  RealVec L2L_U, L2L_V;
  std::vector<RealVec> mat_M2L_Helper;
  std::vector<RealVec> mat_M2M;
  std::vector<RealVec> mat_L2L;

  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);
  RealVec transpose(RealVec& vec, int m, int n);

  void PrecompCheck2Equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    // caculate M2M_U and M2M_V
    RealVec uc_coord = surface(MULTIPOLE_ORDER,c,2.95,level);
    RealVec ue_coord = surface(MULTIPOLE_ORDER,c,1.05,level);
    RealVec M_e2c(NSURF*NSURF);
    kernelMatrix(&ue_coord[0], NSURF, &uc_coord[0], NSURF, &M_e2c[0]);
    RealVec U(NSURF*NSURF), S(NSURF*NSURF), V(NSURF*NSURF);
    svd(NSURF, NSURF, &M_e2c[0], &S[0], &U[0], &V[0]);
    // inverse S
    real_t max_S = 0;
    for(size_t i=0; i<NSURF; i++) {
      max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
    }
    for(size_t i=0; i<NSURF; i++) {
      S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
    }
    // save matrix
    RealVec VT = transpose(V, NSURF, NSURF);
    M2M_V.resize(NSURF*NSURF);
    M2M_U = transpose(U, NSURF, NSURF);
    gemm(NSURF, NSURF, NSURF, &VT[0], &S[0], &M2M_V[0]);

    L2L_V.resize(NSURF*NSURF);
    L2L_U = V;
    gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &L2L_V[0]);
  }

  void PrecompM2M() {
    int level = 0;
    real_t parent_coord[3] = {0, 0, 0};
    RealVec p_check_surf = surface(MULTIPOLE_ORDER,parent_coord,2.95,level);
    real_t s = powf(0.5, level+2);

    int numRelCoord = rel_coord[M2M_Type].size();
    mat_M2M.resize(numRelCoord);
    mat_L2L.resize(numRelCoord);
#pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      ivec3& coord = rel_coord[M2M_Type][i];
      real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
      RealVec c_equiv_surf = surface(MULTIPOLE_ORDER,child_coord,1.05,level+1);
      RealVec M_e2c(NSURF*NSURF);
      kernelMatrix(&c_equiv_surf[0], NSURF, &p_check_surf[0], NSURF, &M_e2c[0]);
      // M2M: child's upward_equiv to parent's check
      RealVec buffer(NSURF*NSURF);
      mat_M2M[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &M_e2c[0], &M2M_V[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &buffer[0], &M2M_U[0], &(mat_M2M[i][0]));
      // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
      M_e2c = transpose(M_e2c, NSURF, NSURF);
      mat_L2L[i].resize(NSURF*NSURF);
      gemm(NSURF, NSURF, NSURF, &L2L_U[0], &M_e2c[0], &buffer[0]);
      gemm(NSURF, NSURF, NSURF, &L2L_V[0], &buffer[0], &(mat_L2L[i][0]));
    }
  }
  
  void PrecompM2LHelper() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, &fftw_in[0], NULL, 1, n3,
                    (fft_complex*)(&fftw_out[0]), NULL, 1, n3_, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L_Helper.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = rel_coord[M2L_Helper_Type][i][d];
      }
      RealVec conv_coord = conv_grid(coord, 0);
      RealVec r_trg(3, 0.0);
      RealVec conv_poten(n3);
      kernelMatrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
      mat_M2L_Helper[i].resize(2*n3_);
      fft_execute_dft_r2c(plan, &conv_poten[0], (fft_complex*)(&mat_M2L_Helper[i][0]));
      // scaling
      for(int k=0; k<2*n3_; ++k) {
        mat_M2L_Helper[i][k] /= n3;
      }
    }
    fft_destroy_plan(plan);
  }


  void Precompute() {
    PrecompCheck2Equiv();
    PrecompM2M();
    PrecompM2LHelper();
  }
}//end namespace
#endif
