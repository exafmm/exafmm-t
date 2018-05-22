#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_
#include "pvfmm.h"
#include "kernel.hpp"
#include "interac_list.hpp"
#include "geometry.h"
#include "profile.hpp"
namespace pvfmm {
  // This is only related to M2M and L2L operator
  void Perm_R(Mat_Type type, size_t indx, const Kernel* kernel) {
    Matrix<real_t>& M0 = (type == M2M_Type) ? mat_M2M : mat_L2L;
    Permutation<real_t>& row_perm = (type == M2M_Type) ? kernel->k_m2m->perm_r[indx] : kernel->k_l2l->perm_r[indx];
    if(row_perm.Dim()==0) { // if this perm_r entry hasn't been computed
      std::vector<Perm_Type> p_list = perm_list[type][indx]; // get perm_list of current rel_coord
      // for(int i=0; i<l; i++) p_list.push_back(Scaling); // push back Scaling operation l times
      Permutation<real_t> row_perm_=Permutation<real_t>(M0.Dim(0)); // init row_perm to be size npts*src_dim
      for(int i=0; i<C_Perm; i++) { // loop over permutation types
        Permutation<real_t>& pr = (type == M2M_Type) ? perm_M2M[i] : perm_L2L[i]; // grab the handle of its mat->perm entry
        if(!pr.Dim()) row_perm_ = Permutation<real_t> (0); // if PrecompPerm never called for this type and entry: this entry does not need permutation so set it empty
      }
      if(row_perm_.Dim()>0) // if this type & entry needs permutation
        for(int i=p_list.size()-1; i>=0; i--) { // loop over the operations of perm_list from end to begin
          //assert(type!=M2L_Helper_Type);
          Permutation<real_t>& pr = (type == M2M_Type) ? perm_M2M[p_list[i]] : perm_L2L[p_list[i]]; // grab the handle of its mat->perm entry
          row_perm_ = pr.Transpose() * row_perm_; // accumulate the permutation to row_perm (perm_r in precompmat header)
        }
      row_perm=row_perm_;
    }
  }

  void Perm_C(Mat_Type type, size_t indx, const Kernel* kernel) {
    Matrix<real_t>& M0 = (type == M2M_Type) ? mat_M2M : mat_L2L;
    Permutation<real_t>& col_perm = (type == M2M_Type) ? kernel->k_m2m->perm_c[indx] : kernel->k_l2l->perm_c[indx];
    if(col_perm.Dim()==0) {
      std::vector<Perm_Type> p_list = perm_list[type][indx];
      Permutation<real_t> col_perm_ = Permutation<real_t>(M0.Dim(1));
      for(int i=0; i<C_Perm; i++) {
        Permutation<real_t>& pc = (type == M2M_Type) ? perm_M2M[C_Perm + i] : perm_L2L[C_Perm + i];
        if(!pc.Dim()) col_perm_ = Permutation<real_t>(0);
      }
      if(col_perm_.Dim()>0)
        for(int i=p_list.size()-1; i>=0; i--) {
          Permutation<real_t>& pc = (type == M2M_Type) ? perm_M2M[C_Perm + p_list[i]] : perm_L2L[C_Perm + p_list[i]];
          col_perm_ = col_perm_*pc;
        }
      col_perm = col_perm_;
    }
  }

  void PrecompPerm(Mat_Type type, Perm_Type perm_indx, const Kernel* kernel) {
    size_t p_indx=perm_indx % C_Perm;
    Permutation<real_t> P;
    switch (type) {
    case M2M_Type: {
      std::vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm) {
        ker_perm=kernel->k_m2m->perm_vec[0+p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      } else {
        ker_perm=kernel->k_m2m->perm_vec[0+p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      }
      perm_M2M[perm_indx] = equiv_surf_perm(p_indx, ker_perm, scal_exp);
      break;
    }
    case L2L_Type: {
      std::vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm) {
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
        for(size_t i=0; i<scal_exp.size(); i++) scal_exp[i]=-scal_exp[i];
      } else {
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
      }
      perm_L2L[perm_indx] = equiv_surf_perm(p_indx, ker_perm, scal_exp);
      break;
    }
    default:
      break;
    }
  }

  void Precomp(Mat_Type type, size_t mat_indx, const Kernel* kernel) {
    int level = 0;
    Matrix<real_t> M;
    switch (type) {
    case M2M_Type: {
      const int* ker_dim=kernel->k_m2m->ker_dim;
      real_t c[3]= {0, 0, 0};
      std::vector<real_t> check_surf=u_check_surf(c, level);
      real_t s=powf(0.5, (level+2));
      ivec3& coord = rel_coord[type][mat_indx];
      real_t child_coord[3]= {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
      std::vector<real_t> equiv_surf=u_equiv_surf(child_coord, level+1);
      Matrix<real_t> M_ce2c(NSURF*ker_dim[0], NSURF*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&equiv_surf[0], NSURF,
                                 &check_surf[0], NSURF, &(M_ce2c[0][0]));
      // caculate M2M_U and M2M_V
      Matrix<real_t> M_c2e0, M_c2e1;
      std::vector<real_t> uc_coord=u_check_surf(c, level);
      std::vector<real_t> ue_coord=u_equiv_surf(c, level);
      Matrix<real_t> M_e2c(NSURF*ker_dim[0], NSURF*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&ue_coord[0], NSURF, &uc_coord[0], NSURF, &(M_e2c[0][0]));
      Matrix<real_t> U, S, V;
      Profile::Tic("SVD", false, 4);
      M_e2c.SVD(U, S, V);
      Profile::Toc();
      real_t eps=1, max_S=0;
      while(eps*(real_t)0.5+(real_t)1.0>1.0) eps*=0.5;
      for(size_t i=0; i<std::min(S.Dim(0), S.Dim(1)); i++) {
        if(fabs(S[i][i])>max_S) max_S=fabs(S[i][i]);
      }
      for(size_t i=0; i<S.Dim(0); i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
      M2M_V = V.Transpose()*S;
      M2M_U = U.Transpose();
      L2L_V = U*S;
      L2L_U = V;

      mat_M2M = (M_ce2c * M2M_V) * M2M_U;
      break;
    }
    case L2L_Type: {
      const int* ker_dim=kernel->k_l2l->ker_dim;
      real_t s=powf(0.5, level+1);
      ivec3& coord=rel_coord[type][mat_indx];
      real_t c[3]= {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
      std::vector<real_t> check_surf=d_check_surf(c, level);
      real_t parent_coord[3]= {0, 0, 0};
      std::vector<real_t> equiv_surf=d_equiv_surf(parent_coord, level-1);
      Matrix<real_t> M_pe2c(NSURF*ker_dim[0], NSURF*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], NSURF, &check_surf[0], NSURF, &(M_pe2c[0][0]));

      Matrix<real_t> M_c2e0, M_c2e1;
      Permutation<real_t> ker_perm=kernel->k_l2l->perm_vec[C_Perm+Scaling];
      std::vector<real_t> scal_exp=kernel->k_l2l->trg_scal;
      Permutation<real_t> P=equiv_surf_perm(Scaling, ker_perm, scal_exp);
      M_c2e0 = P * L2L_V;
      ker_perm=kernel->k_l2l->perm_vec[0     +Scaling];
      scal_exp=kernel->k_l2l->src_scal;
      P=equiv_surf_perm(Scaling, ker_perm, scal_exp);
      M_c2e1 = L2L_U * P;
      mat_L2L = M_c2e0 * (M_c2e1*M_pe2c);
      break;
    }
    case M2L_Helper_Type: {
      const int* ker_dim=kernel->k_m2l->ker_dim;
      int n1=MULTIPOLE_ORDER*2;
      int n3 =n1*n1*n1;
      int n3_=n1*n1*(n1/2+1);
      real_t s=powf(0.5, level);
      ivec3& coord2=rel_coord[type][mat_indx];
      real_t coord_diff[3]= {coord2[0]*s, coord2[1]*s, coord2[2]*s};
      std::vector<real_t> r_trg(3, 0.0);
      std::vector<real_t> conv_poten(n3*ker_dim[0]*ker_dim[1]);
      std::vector<real_t> conv_coord=conv_grid(coord_diff, level);
      kernel->k_m2l->BuildMatrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
      Matrix<real_t> M_conv(n3, ker_dim[0]*ker_dim[1], &conv_poten[0], false);
      M_conv=M_conv.Transpose();
      int err, nnn[3]= {n1, n1, n1};
      real_t *fftw_in, *fftw_out;
      err = posix_memalign((void**)&fftw_in, MEM_ALIGN,   n3 *ker_dim[0]*ker_dim[1]*sizeof(real_t));
      err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*ker_dim[0]*ker_dim[1]*sizeof(real_t));

      if (!m2l_precomp_fft_flag) {
        m2l_precomp_fftplan = fft_plan_many_dft_r2c(3, nnn, ker_dim[0]*ker_dim[1],
                              (real_t*)fftw_in, NULL, 1, n3,
                              (fft_complex*) fftw_out, NULL, 1, n3_,
                              FFTW_ESTIMATE);
        m2l_precomp_fft_flag=true;
      }
      memcpy(fftw_in, &conv_poten[0], n3*ker_dim[0]*ker_dim[1]*sizeof(real_t));
      fft_execute_dft_r2c(m2l_precomp_fftplan, (real_t*)fftw_in, (fft_complex*)(fftw_out));
      Matrix<real_t> M_(2*n3_*ker_dim[0]*ker_dim[1], 1, (real_t*)fftw_out, false);
      mat_M2L_Helper[mat_indx] = M_;
      free(fftw_in);
      free(fftw_out);
      break;
    }
    case M2L_Type: {
      const int* ker_dim=kernel->k_m2l->ker_dim;
      size_t mat_cnt =rel_coord[M2L_Helper_Type].size();
      const size_t chld_cnt=1UL<<3;
      size_t n1=MULTIPOLE_ORDER*2;
      size_t M_dim=n1*n1*(n1/2+1);
      size_t n3=n1*n1*n1;
      std::vector<real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2, 0);
      std::vector<real_t*> M_ptr(chld_cnt*chld_cnt);
      for(size_t i=0; i<chld_cnt*chld_cnt; i++) M_ptr[i]=&zero_vec[0];
      ivec3& rel_coord_=rel_coord[M2L_Type][mat_indx];
      for(int j1=0; j1<chld_cnt; j1++)
        for(int j2=0; j2<chld_cnt; j2++) {
          int relCoord[3]= {rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                             rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                             rel_coord_[2]*2-(j1/4)%2+(j2/4)%2
                            };
          for(size_t k=0; k<mat_cnt; k++) {
            ivec3& ref_coord = rel_coord[M2L_Helper_Type][k];
            if(ref_coord[0] == relCoord[0] &&
                ref_coord[1] == relCoord[1] &&
                ref_coord[2] == relCoord[2]) {
              M_ptr[j2*chld_cnt+j1]= &mat_M2L_Helper[k][0][0];
              break;
            }
          }
        }
      M.Resize(ker_dim[0]*ker_dim[1]*M_dim, 2*chld_cnt*chld_cnt);
      for(int j=0; j<ker_dim[0]*ker_dim[1]*M_dim; j++) {
        for(size_t k=0; k<chld_cnt*chld_cnt; k++) {
          M[j][k*2+0]=M_ptr[k][j*2+0]/n3;
          M[j][k*2+1]=M_ptr[k][j*2+1]/n3;
        }
      }

      mat_M2L[mat_indx] = M;
      break;
    }
    default:
      break;
    }
  }

  void PrecompAll(Mat_Type type, const Kernel* kernel) {
    int idx_num = rel_coord[type].size(); // num of relative pts (rel_coord) w.r.t this type
    if (type == M2M_Type || type == L2L_Type) {
      for(int perm_idx=0; perm_idx<Perm_Count; perm_idx++) PrecompPerm(type, (Perm_Type) perm_idx, kernel);
      for(int i=0; i<idx_num; i++) {           // i is index of rel_coord
        if(interac_class[type][i] == i) { // if i-th coord is a class_coord
          Precomp(type, i, kernel);                       // calculate operator matrix of class_coord
        }
      }
      for(int mat_idx=0; mat_idx<idx_num; mat_idx++) {
        Perm_R(type, mat_idx, kernel);
        Perm_C(type, mat_idx, kernel);
      }
    } else {
      for(int mat_idx=0; mat_idx<idx_num; mat_idx++)
        Precomp(type, mat_idx, kernel);
    }
  }

  void PrecompMat(const Kernel* kernel) {
    perm_M2M.resize(Perm_Count);
    perm_L2L.resize(Perm_Count);
    mat_M2L.resize(rel_coord[M2L_Type].size());
    mat_M2L_Helper.resize(rel_coord[M2L_Helper_Type].size());
    int numRelCoords = rel_coord[M2M_Type].size();
    kernel->k_m2m->perm_r.resize(numRelCoords);
    kernel->k_m2m->perm_c.resize(numRelCoords);
    kernel->k_l2l->perm_r.resize(numRelCoords);
    kernel->k_l2l->perm_c.resize(numRelCoords);
    PrecompAll(M2M_Type, kernel);
    PrecompAll(M2L_Helper_Type, kernel);
    PrecompAll(M2L_Type, kernel);
    PrecompAll(L2L_Type, kernel);
  }

}//end namespace

#endif //_PrecompMAT_HPP_
