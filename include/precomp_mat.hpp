#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_
#include "pvfmm.h"
#include "kernel.hpp"
#include "interac_list.hpp"
#include "geometry.h"
#include "profile.hpp"
namespace pvfmm {
class PrecompMat {
 public:
  std::vector<std::vector<Matrix<real_t> > > mat;
  std::vector<std::vector<Permutation<real_t> > > perm;
  std::vector<std::vector<Permutation<real_t> > > perm_r;
  std::vector<std::vector<Permutation<real_t> > > perm_c;
  InteracList* interacList;
  const Kernel* kernel;

  PrecompMat(InteracList* interacList_, const Kernel* kernel_):
    kernel(kernel_), interacList(interacList_) {
    mat.resize(PrecomputationType);
    for(int type=0; type<PrecomputationType; type++) {
      int numRelCoords = interacList->rel_coord[type].size();
      mat[type].resize(numRelCoords);
    }
    perm.resize(Type_Count);
    perm[M2M_Type].resize(Perm_Count);
    perm[L2L_Type].resize(Perm_Count);
    perm_r.resize(Type_Count);
    perm_c.resize(Type_Count);
    int numRelCoords = interacList->rel_coord[M2M_Type].size();
    perm_r[M2M_Type].resize(numRelCoords);
    perm_c[M2M_Type].resize(numRelCoords);
    perm_r[L2L_Type].resize(numRelCoords);
    perm_c[L2L_Type].resize(numRelCoords);
    PrecompAll(M2M_Type);
    PrecompAll(M2L_Type);
    PrecompAll(L2L_Type);
  }

  // This is only related to M2M and L2L operator
  Permutation<real_t>& Perm_R(int l, Mat_Type type, size_t indx) {
    size_t indx0 =
      interacList->interac_class[type][indx];                     // indx0: class coord index
    Matrix     <real_t>& M0      = mat[type][indx0];         // class coord matrix
    Permutation<real_t>& row_perm = perm_r[l*Type_Count
                                           +type][indx];    // mat->perm_r[(l+128)*16+type][indx]
    //if(M0.Dim(0)==0 || M0.Dim(1)==0) return row_perm;             // if mat hasn't been computed, then return
    if(row_perm.Dim()==0) {                                       // if this perm_r entry hasn't been computed
      std::vector<Perm_Type> p_list =
        interacList->perm_list[type][indx];      // get perm_list of current rel_coord
      for(int i=0; i<l; i++) p_list.push_back(Scaling);           // push back Scaling operation l times
      Permutation<real_t> row_perm_=Permutation<real_t>(M0.Dim(
                                      0));  // init row_perm to be size npts*src_dim
      for(int i=0; i<C_Perm; i++) {                               // loop over permutation types
        Permutation<real_t>& pr = perm[type][R_Perm + i];      // grab the handle of its mat->perm entry
        if(!pr.Dim()) row_perm_ = Permutation<real_t> (0);           // if PrecompPerm never called for this type and entry: this entry does not need permutation so set it empty
      }
      if(row_perm_.Dim()>0)                                      // if this type & entry needs permutation
        for(int i=p_list.size()-1; i>=0; i--) {                   // loop over the operations of perm_list from end to begin
          //assert(type!=M2L_Helper_Type);
          Permutation<real_t>& pr = perm[type][R_Perm + p_list[i]];  // get the permutation of the operation
          row_perm_=pr.Transpose()
                    *row_perm_;                     // accumulate the permutation to row_perm (perm_r in precompmat header)
        }
      row_perm=row_perm_;
    }
    return row_perm;
  }

  Permutation<real_t>& Perm_C(int l, Mat_Type type, size_t indx) {
    size_t indx0 = interacList->interac_class[type][indx];
    Matrix     <real_t>& M0      = mat[type][indx0];
    Permutation<real_t>& col_perm = perm_c[l*Type_Count+type][indx];
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return col_perm;
    if(col_perm.Dim()==0) {
      std::vector<Perm_Type> p_list = interacList->perm_list[type][indx];
      for(int i=0; i<l; i++) p_list.push_back(Scaling);
      Permutation<real_t> col_perm_ = Permutation<real_t>(M0.Dim(1));
      for(int i=0; i<C_Perm; i++) {
        Permutation<real_t>& pc = perm[type][C_Perm + i];
        if(!pc.Dim()) col_perm_ = Permutation<real_t>(0);
      }
      if(col_perm_.Dim()>0)
        for(int i=p_list.size()-1; i>=0; i--) {
          Permutation<real_t>& pc = perm[type][C_Perm + p_list[i]];
          col_perm_ = col_perm_*pc;
        }
      col_perm = col_perm_;
    }
    return col_perm;
  }

  Matrix<real_t>& ClassMat(Mat_Type type, size_t indx) {
    size_t indx0 = interacList->interac_class[type][indx];
    return mat[type][indx0];
  }

  void PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<real_t>& P_ = perm[type][perm_indx];
    if(P_.Dim()!=0) return;
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
      P=equiv_surf_perm(p_indx, ker_perm, scal_exp);
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
      P=equiv_surf_perm(p_indx, ker_perm, scal_exp);
      break;
    }
    default:
      break;
    }
    if(P_.Dim()==0) P_=P;
  }

  Matrix<real_t>& Precomp(Mat_Type type, size_t mat_indx) {
    int level = 0;
    Matrix<real_t>& M_ = mat[type][mat_indx];
    if(M_.Dim(0)!=0 && M_.Dim(1)!=0) return M_;
    Matrix<real_t> M;
    switch (type) {
    case M2M_Type: {
      const int* ker_dim=kernel->k_m2m->ker_dim;
      real_t c[3]= {0, 0, 0};
      std::vector<real_t> check_surf=u_check_surf(c, level);
      real_t s=powf(0.5, (level+2));
      ivec3& coord = interacList->rel_coord[type][mat_indx];
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
      M_c2e0=V.Transpose()*S;
      M_c2e1=U.Transpose();
      mat[M2M_V_Type][0] = V.Transpose()*S;
      mat[M2M_U_Type][0] = U.Transpose();
      mat[L2L_V_Type][0] = U*S;
      mat[L2L_U_Type][0] = V;

Profile::Tic("Multiply Matrix", false, 4);
      M=(M_ce2c*M_c2e0)*M_c2e1;
Profile::Toc();
      break;
    }
    case L2L_Type: {
      const int* ker_dim=kernel->k_l2l->ker_dim;
      real_t s=powf(0.5, level+1);
      ivec3& coord=interacList->rel_coord[type][mat_indx];
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
      M_c2e0=P*mat[L2L_V_Type][0];
      ker_perm=kernel->k_l2l->perm_vec[0     +Scaling];
      scal_exp=kernel->k_l2l->src_scal;
      P=equiv_surf_perm(Scaling, ker_perm, scal_exp);
      M_c2e1=mat[L2L_U_Type][0]*P;
      M=M_c2e0*(M_c2e1*M_pe2c);
      break;
    }
    case M2L_Helper_Type: {
      const int* ker_dim=kernel->k_m2l->ker_dim;
      int n1=MULTIPOLE_ORDER*2;
      int n3 =n1*n1*n1;
      int n3_=n1*n1*(n1/2+1);
      real_t s=powf(0.5, level);
      ivec3& coord2=interacList->rel_coord[type][mat_indx];
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
      M=M_;
      free(fftw_in);
      free(fftw_out);
      break;
    }
    case M2L_Type: {
      const int* ker_dim=kernel->k_m2l->ker_dim;
      size_t mat_cnt =interacList->rel_coord[M2L_Helper_Type].size();
      for(size_t k=0; k<mat_cnt; k++) Precomp(M2L_Helper_Type, k);
      const size_t chld_cnt=1UL<<3;
      size_t n1=MULTIPOLE_ORDER*2;
      size_t M_dim=n1*n1*(n1/2+1);
      size_t n3=n1*n1*n1;
      std::vector<real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2, 0);
      std::vector<real_t*> M_ptr(chld_cnt*chld_cnt);
      for(size_t i=0; i<chld_cnt*chld_cnt; i++) M_ptr[i]=&zero_vec[0];
      ivec3& rel_coord_=interacList->rel_coord[M2L_Type][mat_indx];
      for(int j1=0; j1<chld_cnt; j1++)
        for(int j2=0; j2<chld_cnt; j2++) {
          int rel_coord[3]= {rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                             rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                             rel_coord_[2]*2-(j1/4)%2+(j2/4)%2
                            };
          for(size_t k=0; k<mat_cnt; k++) {
            ivec3& ref_coord=interacList->rel_coord[M2L_Helper_Type][k];
            if(ref_coord[0]==rel_coord[0] &&
                ref_coord[1]==rel_coord[1] &&
                ref_coord[2]==rel_coord[2]) {
              Matrix<real_t>& M = mat[M2L_Helper_Type][k];
              M_ptr[j2*chld_cnt+j1]=&M[0][0];
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
      break;
    }
    default:
      break;
    }
    if(M_.Dim(0)==0 && M_.Dim(1)==0)
      M_=M;
    return M_;
  }

  void PrecompAll(Mat_Type type) {
    int idx_num = interacList->rel_coord[type].size(); // num of relative pts (rel_coord) w.r.t this type
    if (type == M2M_Type || type == L2L_Type) {
      for(int perm_idx=0; perm_idx<Perm_Count; perm_idx++) PrecompPerm(type, (Perm_Type) perm_idx);
      for(int i=0; i<idx_num; i++) {           // i is index of rel_coord
        if(interacList->interac_class[type][i] == i) { // if i-th coord is a class_coord
          Precomp(type, i);                       // calculate operator matrix of class_coord
        }
      }
      for(int mat_idx=0; mat_idx<idx_num; mat_idx++) {
        Perm_R(0, type, mat_idx);
        Perm_C(0, type, mat_idx);
      }
    } else {
      for(int mat_idx=0; mat_idx<idx_num; mat_idx++)
        Precomp(type, mat_idx);
    }
  }
};

}//end namespace

#endif //_PrecompMAT_HPP_
