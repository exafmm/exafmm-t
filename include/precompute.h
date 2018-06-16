#ifndef precompute_h
#define precompute_h
#include "exafmm_t.h"
#include "geometry.h"
#include "kernel.h"

namespace exafmm_t {
  void PrecompPerm() {
    perm_M2M.resize(Perm_Count);
    Permutation<real_t> ker_perm(1);
    #pragma omp parallel for
    for(int p=0; p<Perm_Count; p++) {
      size_t p_indx = p % C_Perm;
      perm_M2M[p] = equiv_surf_perm(p_indx, ker_perm, 0);
    }
  }

  void Perm_R() {
    int numRelCoord = rel_coord[M2M_Type].size();
    perm_r.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      std::vector<Perm_Type> p_list = perm_list[M2M_Type][i];   // get perm_list of current rel_coord
      Permutation<real_t> row_perm_ = Permutation<real_t>(NSURF); // init row_perm to be size NSURF
      for(int j=p_list.size()-1; j>=0; j--) { // loop over the operations of perm_list from end to begin
        Permutation<real_t>& pr = perm_M2M[p_list[j]]; // grab the handle of its mat->perm entry
        row_perm_ = pr.Transpose() * row_perm_; // accumulate the permutation to row_perm (perm_r in precompmat header)
      }
      perm_r[i] = row_perm_;
    }
  }

  void Perm_C() {
    int numRelCoord = rel_coord[M2M_Type].size();
    perm_c.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      std::vector<Perm_Type> p_list = perm_list[M2M_Type][i];
      Permutation<real_t> col_perm_ = Permutation<real_t>(NSURF);
        for(int j=p_list.size()-1; j>=0; j--) {
          Permutation<real_t>& pc = perm_M2M[C_Perm + p_list[j]];
          col_perm_ = col_perm_ * pc;
        }
      perm_c[i] = col_perm_;
    }
  }

  void PrecompM2M() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    RealVec check_surf = u_check_surf(c, level);
    real_t s = powf(0.5, level+2);
    int class_coord_idx = interac_class[M2M_Type][0];
    ivec3& coord = rel_coord[M2M_Type][class_coord_idx];
    real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
    RealVec equiv_surf = u_equiv_surf(child_coord, level+1);
    RealVec M_ce2c(NSURF*NSURF);
    BuildMatrix(&equiv_surf[0], NSURF, &check_surf[0], NSURF, &M_ce2c[0]);
    // caculate M2M_U and M2M_V
    RealVec uc_coord = u_check_surf(c, level);
    RealVec ue_coord = u_equiv_surf(c, level);
    RealVec M_e2c(NSURF*NSURF);
    BuildMatrix(&ue_coord[0], NSURF, &uc_coord[0], NSURF, &M_e2c[0]);
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

    RealVec VT = transpose(V, NSURF, NSURF);
    M2M_V.resize(NSURF*NSURF);
    M2M_U = transpose(U, NSURF, NSURF);
    gemm(NSURF, NSURF, NSURF, &VT[0], &S[0], &M2M_V[0]);

    L2L_V.resize(NSURF*NSURF);
    L2L_U = V;
    gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &L2L_V[0]);

    mat_M2M.resize(NSURF*NSURF);
    RealVec buffer(NSURF*NSURF);
    gemm(NSURF, NSURF, NSURF, &M_ce2c[0], &M2M_V[0], &buffer[0]);
    gemm(NSURF, NSURF, NSURF, &buffer[0], &M2M_U[0], &mat_M2M[0]);
  }

  void PrecompL2L() {
    int class_coord_idx = interac_class[L2L_Type][0];
    int level = 0;
    real_t s = powf(0.5, level+1);
    ivec3& coord = rel_coord[L2L_Type][class_coord_idx];
    real_t c[3]= {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
    RealVec check_surf = d_check_surf(c, level);
    real_t parent_coord[3] = {0, 0, 0};
    RealVec equiv_surf = d_equiv_surf(parent_coord, level-1);
    RealVec M_pe2c(NSURF*NSURF);
    BuildMatrix(&equiv_surf[0], NSURF, &check_surf[0], NSURF, &M_pe2c[0]);

    mat_L2L.resize(NSURF*NSURF);
    RealVec buffer(NSURF*NSURF);
    gemm(NSURF, NSURF, NSURF, &L2L_U[0], &M_pe2c[0], &buffer[0]);
    gemm(NSURF, NSURF, NSURF, &L2L_V[0], &buffer[0], &mat_L2L[0]);
  }

  void PrecompM2LHelper() {
    // create fftw plan
    RealVec fftw_in(N3);
    RealVec fftw_out(2*N3_);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, &fftw_in[0], NULL, 1, N3,
                    (fft_complex*)(&fftw_out[0]), NULL, 1, N3_, FFTW_ESTIMATE);
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
      RealVec conv_poten(N3);
      BuildMatrix(&conv_coord[0], N3, &r_trg[0], 1, &conv_poten[0]);
      mat_M2L_Helper[i].resize(2*N3_);
      fft_execute_dft_r2c(plan, &conv_poten[0], (fft_complex*)(&mat_M2L_Helper[i][0]));
    }
    fft_destroy_plan(plan);
  }

  void PrecompM2L() {
    int numParentRelCoord = rel_coord[M2L_Type].size();
    int numChildRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L.resize(numParentRelCoord);
    RealVec zero_vec(N3_*2, 0);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<numParentRelCoord; i++) {
      ivec3& parentRelCoord = rel_coord[M2L_Type][i];
      std::vector<real_t*> M_ptr(NCHILD*NCHILD, &zero_vec[0]);
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
              M_ptr[j2*NCHILD+j1] = &mat_M2L_Helper[k][0];
              break;
            }
          }
        }
      }
      mat_M2L[i].resize(N3_*2*NCHILD*NCHILD);  // N3 by (2*NCHILD*NCHILD) matrix
      for(int k=0; k<N3_; k++) {                      // loop over frequencies
        for(size_t j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int index = k*(2*NCHILD*NCHILD) + 2*j;
          mat_M2L[i][index+0] = M_ptr[j][k*2+0]/N3;   // real
          mat_M2L[i][index+1] = M_ptr[j][k*2+1]/N3;   // imag
        }
      }
    }
  }

  void Precompute() {
    PrecompPerm();
    Perm_R();
    Perm_C();
    PrecompM2M();
    PrecompL2L();
    PrecompM2LHelper();
    PrecompM2L();
  }
}//end namespace
#endif
