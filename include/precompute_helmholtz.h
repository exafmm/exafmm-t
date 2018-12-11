#ifndef precompute_helmholtz_h
#define precompute_helmholtz_h
#include "exafmm_t.h"
#include "geometry.h"
#include "helmholtz.h"

namespace exafmm_t {
  std::vector<ComplexVec> M2M_U, M2M_V;
  std::vector<ComplexVec> L2L_U, L2L_V;
  std::vector<std::vector<ComplexVec>> mat_M2M, mat_L2L;
  std::vector<std::vector<RealVec>> mat_M2L_Helper;
  std::vector<std::vector<RealVec>> mat_M2L;

  void PrecompCheck2Equiv() {
    real_t c[3] = {0, 0, 0};
    M2M_V.resize(MAXLEVEL+1);
    M2M_U.resize(MAXLEVEL+1);
    L2L_V.resize(MAXLEVEL+1);
    L2L_U.resize(MAXLEVEL+1);
    for(int level = 0; level <= MAXLEVEL; level++) {
      // caculate M2M_U and M2M_V
      RealVec up_check_surf = surface(MULTIPOLE_ORDER,c,2.95,level);
      RealVec up_equiv_surf = surface(MULTIPOLE_ORDER,c,1.05,level);
      ComplexVec M_c2e(NSURF*NSURF);
      kernelMatrix(&up_check_surf[0], NSURF, &up_equiv_surf[0], NSURF, &M_c2e[0]);
      RealVec S(NSURF*NSURF);
      ComplexVec U(NSURF*NSURF), VT(NSURF*NSURF);
      svd(NSURF, NSURF, &M_c2e[0], &S[0], &U[0], &VT[0]);
      // inverse S
      real_t max_S = 0;
      for(size_t i=0; i<NSURF; i++) {
        max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
      }
      for(size_t i=0; i<NSURF; i++) {
        S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
      }
      // save matrix
      ComplexVec V = transpose(VT, NSURF, NSURF);
      M2M_V[level].resize(NSURF*NSURF);
      M2M_U[level] = transpose(U, NSURF, NSURF);
      gemm(NSURF, NSURF, NSURF, &V[0], &S[0], &(M2M_V[level][0]));

      L2L_V[level].resize(NSURF*NSURF);
      L2L_U[level] = VT;
      gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &(L2L_V[level][0]));
#if 0
      // check M2M_U, M2M_V, L2L_U, L2L_V
      std::cout << "level: " << level << std::endl;
      for(int i = 0; i < NSURF*NSURF; i++) {
        std::cout << M2M_U[level][i] << " , " << M2M_V[level][i] << " , " << L2L_U[level][i] << " , " << L2L_V[level][i] << std::endl;
      }
#endif
    }
  }

  void PrecompM2M() {
    mat_M2M.resize(MAXLEVEL+1);
    mat_L2L.resize(MAXLEVEL+1);
    real_t parent_coord[3] = {0, 0, 0};
    for(int level = 0; level <= MAXLEVEL; level++) {
      RealVec parent_up_check_surf = surface(MULTIPOLE_ORDER,parent_coord,2.95,level);
      real_t s = R0 * powf(0.5, level+1);

      int numRelCoord = rel_coord[M2M_Type].size();
      mat_M2M[level].resize(numRelCoord);
      mat_L2L[level].resize(numRelCoord);
#pragma omp parallel for
      for(int i=0; i<numRelCoord; i++) {
        ivec3& coord = rel_coord[M2M_Type][i];
        real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
        RealVec child_up_equiv_surf = surface(MULTIPOLE_ORDER,child_coord,1.05,level+1);
        ComplexVec M_pc2ce(NSURF*NSURF);
        kernelMatrix(&parent_up_check_surf[0], NSURF, &child_up_equiv_surf[0], NSURF, &M_pc2ce[0]);
        // M2M: child's upward_equiv to parent's check
        ComplexVec buffer(NSURF*NSURF);
        mat_M2M[level][i].resize(NSURF*NSURF);
        gemm(NSURF, NSURF, NSURF, &(M2M_U[level][0]), &M_pc2ce[0], &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &(M2M_V[level][0]), &buffer[0], &(mat_M2M[level][i][0]));
        // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
        M_pc2ce = transpose(M_pc2ce, NSURF, NSURF);
        mat_L2L[level][i].resize(NSURF*NSURF);
        gemm(NSURF, NSURF, NSURF, &M_pc2ce[0], &(L2L_V[level][0]), &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &buffer[0], &(L2L_U[level][0]), &(mat_L2L[level][i][0]));
      }
    }
  }

  void PrecompM2LHelper() {
    mat_M2L_Helper.resize(MAXLEVEL+1);
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    // create fftw plan
    ComplexVec fftw_in(n3);
    RealVec fftw_out(2*n3);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_dft(3, dim, reinterpret_cast<fft_complex*>(&fftw_in[0]), 
                                (fft_complex*)(&fftw_out[0]), FFTW_FORWARD, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = rel_coord[M2L_Helper_Type].size();
    RealVec r_trg(3, 0.0);
    for(int l=0; l<=MAXLEVEL; l++) {
      mat_M2L_Helper[l].resize(numRelCoord);
      #pragma omp parallel for
      for(int i=0; i<numRelCoord; i++) {
        real_t coord[3];
        for(int d=0; d<3; d++) {
          coord[d] = rel_coord[M2L_Helper_Type][i][d] * R0 * powf(0.5, l-1);
        }
        RealVec conv_coord = conv_grid(coord, l);
        ComplexVec conv_poten(n3);
        kernelMatrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
        mat_M2L_Helper[l][i].resize(2*n3);
        fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&conv_poten[0]), (fft_complex*)(&mat_M2L_Helper[l][i][0]));
      }
    }
    fft_destroy_plan(plan);
  }

  void PrecompM2L() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int numParentRelCoord = rel_coord[M2L_Type].size();
    mat_M2L.resize(MAXLEVEL);  // skip the last level
    // parent rel, child rel -> m2l_helper_idx
    std::vector<std::vector<int>> index_mapping(numParentRelCoord, std::vector<int>(NCHILD*NCHILD));
    for(int i=0; i<numParentRelCoord; ++i) {
      for(int j1=0; j1<NCHILD; ++j1) {
        for(int j2=0; j2<NCHILD; ++j2) {
          ivec3& parent_rel_coord = rel_coord[M2L_Type][i];
          ivec3  child_rel_coord;
          child_rel_coord[0] = parent_rel_coord[0]*2 - (j1/1)%2 + (j2/1)%2;
          child_rel_coord[1] = parent_rel_coord[1]*2 - (j1/2)%2 + (j2/2)%2;
          child_rel_coord[2] = parent_rel_coord[2]*2 - (j1/4)%2 + (j2/4)%2;
          int coord_hash = hash(child_rel_coord);
          int child_rel_idx = hash_lut[M2L_Helper_Type][coord_hash];
          int j = j2*NCHILD + j1;
          index_mapping[i][j] = child_rel_idx;
        }
      }
    }
    // copy from mat_M2L_Helper to mat_M2L
    for(int l=0; l<MAXLEVEL; ++l) {
      mat_M2L[l].resize(numParentRelCoord);
      for(int i=0; i<numParentRelCoord; ++i) {
        mat_M2L[l][i].resize(n3 * 2*NCHILD*NCHILD, 0.);  // N3 by (2*NCHILD*NCHILD) matrix
        for(int j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int child_rel_idx = index_mapping[i][j];
          if (child_rel_idx != -1) {
            for(int k=0; k<n3; k++) {                      // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              mat_M2L[l][i][new_idx+0] = mat_M2L_Helper[l+1][child_rel_idx][k*2+0] / n3;   // real
              mat_M2L[l][i][new_idx+1] = mat_M2L_Helper[l+1][child_rel_idx][k*2+1] / n3;   // imag
            }
          }
        }
      }
    }
  }

  void Precompute() {
    PrecompCheck2Equiv();
    PrecompM2M();
    PrecompM2LHelper();
    PrecompM2L();
  }
}//end namespace
#endif
