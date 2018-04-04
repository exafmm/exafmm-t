#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_
#include "pvfmm.h"
#include "kernel.hpp"
#include "interac_list.hpp"
#include "geometry.h"
#include "profile.hpp"
namespace pvfmm{
class PrecompMat{
public:
  std::vector<std::vector<Matrix<real_t> > > mat;
  std::vector<std::vector<Permutation<real_t> > > perm;
  std::vector<std::vector<Permutation<real_t> > > perm_r;
  std::vector<std::vector<Permutation<real_t> > > perm_c;
  InteracList* interacList;
  int multipole_order;
  const Kernel* kernel;

  PrecompMat(InteracList* interacList_, int multipole_order_, const Kernel* kernel_):
    multipole_order(multipole_order_), kernel(kernel_), interacList(interacList_) {
    mat.resize(PrecomputationType);
    for(int type; type<PrecomputationType; type++) {
      int numRelCoords = interacList->rel_coord[type].size();
      mat[type].resize(numRelCoords);
    }

    perm.resize(Type_Count);
    for(size_t i=0;i<Type_Count;i++){
      perm[i].resize(Perm_Count);
    }

    perm_r.resize(MAX_DEPTH*Type_Count);
    perm_c.resize(MAX_DEPTH*Type_Count);
    int numRelCoords = interacList->rel_coord[M2M_Type].size();
    for(size_t i=0; i<perm_r.size(); i++){
      perm_r[i].resize(numRelCoords);
      perm_c[i].resize(numRelCoords);
    }

    Profile::Tic("InitFMM_Pts",true);

    std::stringstream ss;
    const char* realType = sizeof(real_t) == 8 ? "_d" : "_f";
    ss << "Precomp_" << kernel->ker_name.c_str() << "_m" << multipole_order << realType <<".data";
    std::string fname = ss.str();

    Profile::Tic("LoadMatrices",true,3);
    LoadFile(fname.c_str());
    Profile::Toc();

    Profile::Tic("PrecompM2M",false,4);
    PrecompAll(M2M_Type);
    Profile::Toc();
    Profile::Tic("PrecompL2L",false,4);
    PrecompAll(L2L_Type);
    Profile::Toc();

    Profile::Tic("Save2File",false,4);
    FILE* f = fopen(fname.c_str(),"r");
    if(f==NULL) {
      Save2File(fname.c_str());
    }
    else fclose(f);
    Profile::Toc();

    Profile::Tic("PrecompV",false,4);
    // PrecompAll(M2L_Helper_Type);
    Profile::Toc();
    Profile::Tic("PrecompM2L",false,4);
    PrecompAll(M2L_Type);
    Profile::Toc();

    Profile::Toc();
  }

  Permutation<real_t>& getPerm_R(int l, Mat_Type type, size_t indx){
    return perm_r[l*Type_Count+type][indx];
  }

  Permutation<real_t>& getPerm_C(int l, Mat_Type type, size_t indx){
    return perm_c[l*Type_Count+type][indx];
  }
  
  // This is only related to M2M and L2L operator
  Permutation<real_t>& Perm_R(int l, Mat_Type type, size_t indx){
    size_t indx0 = interacList->interac_class[type][indx];                     // indx0: class coord index
    Matrix     <real_t>& M0      = mat[type][indx0];         // class coord matrix
    Permutation<real_t>& row_perm = getPerm_R(l, type, indx );    // mat->perm_r[(l+128)*16+type][indx]
    //if(M0.Dim(0)==0 || M0.Dim(1)==0) return row_perm;             // if mat hasn't been computed, then return
    if(row_perm.Dim()==0){                                        // if this perm_r entry hasn't been computed
      std::vector<Perm_Type> p_list = interacList->perm_list[type][indx];      // get perm_list of current rel_coord
      for(int i=0;i<l;i++) p_list.push_back(Scaling);             // push back Scaling operation l times
      Permutation<real_t> row_perm_=Permutation<real_t>(M0.Dim(0));  // init row_perm to be size npts*src_dim
      for(int i=0;i<C_Perm;i++){                                  // loop over permutation types
        Permutation<real_t>& pr = perm[type][R_Perm + i];      // grab the handle of its mat->perm entry
        if(!pr.Dim()) row_perm_ = Permutation<real_t>(0);           // if PrecompPerm never called for this type and entry: this entry does not need permutation so set it empty
      }
      if(row_perm_.Dim()>0)                                      // if this type & entry needs permutation
      for(int i=p_list.size()-1; i>=0; i--){                    // loop over the operations of perm_list from end to begin
        //assert(type!=M2L_Helper_Type);
        Permutation<real_t>& pr = perm[type][R_Perm + p_list[i]];  // get the permutation of the operation
        row_perm_=pr.Transpose()*row_perm_;                     // accumulate the permutation to row_perm (perm_r in precompmat header)
      }
      row_perm=row_perm_;
    }
    return row_perm;
  }

  Permutation<real_t>& Perm_C(int l, Mat_Type type, size_t indx){
    size_t indx0 = interacList->interac_class[type][indx];
    Matrix     <real_t>& M0      = mat[type][indx0];
    Permutation<real_t>& col_perm = getPerm_C(l, type, indx );
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return col_perm;
    if(col_perm.Dim()==0){
      std::vector<Perm_Type> p_list = interacList->perm_list[type][indx];
      for(int i=0;i<l;i++) p_list.push_back(Scaling);
      Permutation<real_t> col_perm_ = Permutation<real_t>(M0.Dim(1));
      for(int i=0;i<C_Perm;i++){
        Permutation<real_t>& pc = perm[type][C_Perm + i];
        if(!pc.Dim()) col_perm_ = Permutation<real_t>(0);
      }
      if(col_perm_.Dim()>0)
      for(int i=p_list.size()-1; i>=0; i--){
        //assert(type!=M2L_Helper_Type);
        Permutation<real_t>& pc = perm[type][C_Perm + p_list[i]];
        col_perm_ = col_perm_*pc;
      }
      col_perm = col_perm_;
    }
    return col_perm;
  }

  Matrix<real_t>& ClassMat(Mat_Type type, size_t indx){
    size_t indx0 = interacList->interac_class[type][indx];
    return mat[type][indx0];
  }

  void PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<real_t>& P_ = perm[type][perm_indx];
    if(P_.Dim()!=0) return;
    size_t m=multipole_order;
    size_t p_indx=perm_indx % C_Perm;
    Permutation<real_t> P;
    switch (type) {
    case M2M_Type: {
      std::vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm) {
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      }else{
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, scal_exp);
      break;
    }
    case L2L_Type: {
      std::vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm){
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
        for(size_t i=0; i<scal_exp.size(); i++) scal_exp[i]=-scal_exp[i];
      }else{
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, scal_exp);
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
    switch (type){
    case M2M_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2m->ker_dim;
      real_t c[3]={0,0,0};
      std::vector<real_t> check_surf=u_check_surf(multipole_order,c,level);
      size_t n_uc=check_surf.size()/3;
      real_t s=powf(0.5,(level+2));
      ivec3& coord = interacList->rel_coord[type][mat_indx];
      real_t child_coord[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<real_t> equiv_surf=u_equiv_surf(multipole_order,child_coord,level+1);
      size_t n_ue=equiv_surf.size()/3;
      Matrix<real_t> M_ce2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&equiv_surf[0], n_ue,
                                 &check_surf[0], n_uc, &(M_ce2c[0][0]));
      // caculate M2M_U and M2M_V
      Matrix<real_t> M_c2e0, M_c2e1;
      if(multipole_order != 0) {
        const int* ker_dim=kernel->k_m2m->ker_dim;
        real_t c[3]={0,0,0};
        std::vector<real_t> uc_coord=u_check_surf(multipole_order,c,level);
        size_t n_uc=uc_coord.size()/3;
        std::vector<real_t> ue_coord=u_equiv_surf(multipole_order,c,level);
        size_t n_ue=ue_coord.size()/3;
        Matrix<real_t> M_e2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
        kernel->k_m2m->BuildMatrix(&ue_coord[0], n_ue, &uc_coord[0], n_uc, &(M_e2c[0][0]));
        Matrix<real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        M_c2e1=U.Transpose();
        real_t eps=1, max_S=0;
        while(eps*(real_t)0.5+(real_t)1.0>1.0) eps*=0.5;
        for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
          if(fabs(S[i][i])>max_S) max_S=fabs(S[i][i]);
        }
        for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
        M_c2e0=V.Transpose()*S;
      }
      mat[M2M_V_Type][0] = M_c2e0;
      mat[M2M_U_Type][0] = M_c2e1;
      M=(M_ce2c*M_c2e0)*M_c2e1;
      break;
    }
    case L2L_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_l2l->ker_dim;
      real_t s=powf(0.5,level+1);
      ivec3& coord=interacList->rel_coord[type][mat_indx];
      real_t c[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<real_t> check_surf=d_check_surf(multipole_order,c,level);
      size_t n_dc=check_surf.size()/3;
      real_t parent_coord[3]={0,0,0};
      std::vector<real_t> equiv_surf=d_equiv_surf(multipole_order,parent_coord,level-1);
      size_t n_de=equiv_surf.size()/3;
      Matrix<real_t> M_pe2c(n_de*ker_dim[0],n_dc*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_de, &check_surf[0], n_dc, &(M_pe2c[0][0]));

      // caculate L2L_U and L2L_V
      Matrix<real_t> M_c2e0, M_c2e1;
      if(multipole_order != 0 ) {
        const int* ker_dim=kernel->k_l2l->ker_dim;
        real_t c[3]={0,0,0};
        std::vector<real_t> check_surf=d_check_surf(multipole_order,c,level);
        size_t n_ch=check_surf.size()/3;
        std::vector<real_t> equiv_surf=d_equiv_surf(multipole_order,c,level);
        size_t n_eq=equiv_surf.size()/3;
        Matrix<real_t> M_e2c(n_eq*ker_dim[0],n_ch*ker_dim[1]);
        kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_eq, &check_surf[0], n_ch, &(M_e2c[0][0]));
        Matrix<real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        M_c2e1=U.Transpose();
        real_t eps=1, max_S=0;
        while(eps*(real_t)0.5+(real_t)1.0>1.0) eps*=0.5;
        for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
          if(fabs(S[i][i])>max_S) max_S=fabs(S[i][i]);
        }
        for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
        M_c2e0=V.Transpose()*S;
      } 
      mat[L2L_V_Type][0] = M_c2e0;
      mat[L2L_U_Type][0] = M_c2e1;

      Permutation<real_t> ker_perm=kernel->k_l2l->perm_vec[C_Perm+Scaling];
      std::vector<real_t> scal_exp=kernel->k_l2l->trg_scal;
      Permutation<real_t> P=equiv_surf_perm(multipole_order, Scaling, ker_perm, scal_exp);
      M_c2e0=P*M_c2e0;
      ker_perm=kernel->k_l2l->perm_vec[0     +Scaling];
      scal_exp=kernel->k_l2l->src_scal;
      P=equiv_surf_perm(multipole_order, Scaling, ker_perm, scal_exp);
      M_c2e1=M_c2e1*P;
      M=M_c2e0*(M_c2e1*M_pe2c);
      break;
    }
    case M2L_Helper_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      int n1=multipole_order*2;
      int n3 =n1*n1*n1;
      int n3_=n1*n1*(n1/2+1);
      real_t s=powf(0.5,level);
      ivec3& coord2=interacList->rel_coord[type][mat_indx];
      real_t coord_diff[3]={coord2[0]*s,coord2[1]*s,coord2[2]*s};
      std::vector<real_t> r_trg(3,0.0);
      std::vector<real_t> conv_poten(n3*ker_dim[0]*ker_dim[1]);
      std::vector<real_t> conv_coord=conv_grid(multipole_order,coord_diff,level);
      kernel->k_m2l->BuildMatrix(&conv_coord[0],n3,&r_trg[0],1,&conv_poten[0]);
      Matrix<real_t> M_conv(n3,ker_dim[0]*ker_dim[1],&conv_poten[0],false);
      M_conv=M_conv.Transpose();
      int err, nnn[3]={n1,n1,n1};
      real_t *fftw_in, *fftw_out;
      err = posix_memalign((void**)&fftw_in , MEM_ALIGN,   n3 *ker_dim[0]*ker_dim[1]*sizeof(real_t));
      err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*ker_dim[0]*ker_dim[1]*sizeof(real_t));
#pragma omp critical (FFTW_PLAN)
      {
        if (!m2l_precomp_fft_flag){
          m2l_precomp_fftplan = fft_plan_many_dft_r2c(3, nnn, ker_dim[0]*ker_dim[1],
                                                   (real_t*)fftw_in, NULL, 1, n3,
                                                   (fft_complex*) fftw_out, NULL, 1, n3_,
                                                   FFTW_ESTIMATE);
          m2l_precomp_fft_flag=true;
        }
      }
      memcpy(fftw_in, &conv_poten[0], n3*ker_dim[0]*ker_dim[1]*sizeof(real_t));
      fft_execute_dft_r2c(m2l_precomp_fftplan, (real_t*)fftw_in, (fft_complex*)(fftw_out));
      Matrix<real_t> M_(2*n3_*ker_dim[0]*ker_dim[1],1,(real_t*)fftw_out,false);
      M=M_;
      free(fftw_in);
      free(fftw_out);
      break;
    }
    case M2L_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      size_t mat_cnt =interacList->rel_coord[M2L_Helper_Type].size();
      for(size_t k=0;k<mat_cnt;k++) Precomp(M2L_Helper_Type, k);

      const size_t chld_cnt=1UL<<3;
      size_t n1=multipole_order*2;
      size_t M_dim=n1*n1*(n1/2+1);
      size_t n3=n1*n1*n1;

      std::vector<real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2, 0);
      std::vector<real_t*> M_ptr(chld_cnt*chld_cnt);
      for(size_t i=0;i<chld_cnt*chld_cnt;i++) M_ptr[i]=&zero_vec[0];

      ivec3& rel_coord_=interacList->rel_coord[M2L_Type][mat_indx];
      for(int j1=0;j1<chld_cnt;j1++)
        for(int j2=0;j2<chld_cnt;j2++){
          int rel_coord[3]={rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                            rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                            rel_coord_[2]*2-(j1/4)%2+(j2/4)%2};
          for(size_t k=0;k<mat_cnt;k++){
            ivec3& ref_coord=interacList->rel_coord[M2L_Helper_Type][k];
            if(ref_coord[0]==rel_coord[0] &&
               ref_coord[1]==rel_coord[1] &&
               ref_coord[2]==rel_coord[2]){
              Matrix<real_t>& M = mat[M2L_Helper_Type][k];
              M_ptr[j2*chld_cnt+j1]=&M[0][0];
              break;
            }
          }
        }
      M.Resize(ker_dim[0]*ker_dim[1]*M_dim, 2*chld_cnt*chld_cnt);
      for(int j=0;j<ker_dim[0]*ker_dim[1]*M_dim;j++){
        for(size_t k=0;k<chld_cnt*chld_cnt;k++){
          M[j][k*2+0]=M_ptr[k][j*2+0]/n3;
          M[j][k*2+1]=M_ptr[k][j*2+1]/n3;
        }
      }
      break;
    }
    default:
      break;
    }
#pragma omp critical (PRECOMP_MATRIX_PTS)
    if(M_.Dim(0)==0 && M_.Dim(1)==0){
      M_=M;
    }
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
      // for(int mat_idx=0; mat_idx<mat_cnt; mat_idx++) {
      //   for(int level=0; level<MAX_DEPTH; level++) {
      //     Perm_R(level, type, mat_idx);  // calculate perm_r
      //     Perm_C(level, type, mat_idx);  // & perm_c for M2M and D2U type
      //   }
      // }

      for(int mat_idx=0; mat_idx<idx_num; mat_idx++) {
        Permutation<real_t>& perm_r0 = Perm_R(0, type, mat_idx);
        Permutation<real_t>& perm_c0 = Perm_C(0, type, mat_idx);
        for(int level=1; level<MAX_DEPTH; level++) {
          Permutation<real_t>& temp_r = getPerm_R(level, type, mat_idx);
          Permutation<real_t>& temp_c = getPerm_C(level, type, mat_idx);
          temp_r = perm_r0;
          temp_c = perm_c0;
          if(type == L2L_Type) {
            temp_r.scal = std::vector<real_t>(temp_r.scal.size(), pow(0.5, level));
            temp_c.scal = std::vector<real_t>(temp_c.scal.size(), pow(2, level));
          }
        }
      }
    } else {
      for(int mat_idx=0; mat_idx<idx_num; mat_idx++)
        Precomp(type, mat_idx);
    }
  }

  size_t CompactData(int level, Mat_Type type, std::vector<char>& comp_data){
    std::vector<Matrix<real_t> >& mat_ = mat[type];
    size_t mat_cnt=mat_.size();
    size_t indx_size=0;
    size_t mem_size=0;
    size_t l0=0;
    size_t l1= MAX_DEPTH;
    {
      indx_size+=mat_cnt*(1+(2+2)*(l1-l0))*sizeof(size_t);
      for(size_t j=0;j<mat_cnt;j++){
        Matrix     <real_t>& M = mat[type][j];
        if(M.Dim(0)>0 && M.Dim(1)>0){
          mem_size+=M.Dim(0)*M.Dim(1)*sizeof(real_t);
        }
        for(size_t l=l0;l<l1;l++){
          Permutation<real_t>& Pr=getPerm_R(l,type,j);
          Permutation<real_t>& Pc=getPerm_C(l,type,j);
          assert(Pr.Dim() == Pc.Dim());
          if(Pr.Dim()>0){
            mem_size+=Pr.Dim()*sizeof(size_t);
            mem_size+=Pr.Dim()*sizeof(real_t);
            mem_size+=Pc.Dim()*sizeof(size_t);
            mem_size+=Pc.Dim()*sizeof(real_t);
          }
        }
      }
    }
    if(comp_data.size() < indx_size+mem_size) comp_data.resize(indx_size+mem_size);
    {
      char* indx_ptr=&comp_data[0];
      Matrix<size_t> offset_indx(mat_cnt,1+(2+2)*(l1-l0), (size_t*)indx_ptr, false);
      size_t data_offset = indx_size;
      for(size_t j=0;j<mat_cnt;j++){
        Matrix     <real_t>& M = mat[type][j];
        offset_indx[j][0]=data_offset; indx_ptr+=sizeof(size_t);
        data_offset += M.Dim(0)*M.Dim(1)*sizeof(real_t);
        for(size_t l=l0;l<l1;l++){
          Permutation<real_t>& Pr=getPerm_R(l,type,j);
          offset_indx[j][1+4*(l-l0)+0]=data_offset;
          data_offset+=Pr.Dim()*sizeof(size_t);
          offset_indx[j][1+4*(l-l0)+1]=data_offset;
          data_offset+=Pr.Dim()*sizeof(real_t);
          Permutation<real_t>& Pc=getPerm_C(l,type,j);
          offset_indx[j][1+4*(l-l0)+2]=data_offset;
          data_offset+=Pc.Dim()*sizeof(size_t);
          offset_indx[j][1+4*(l-l0)+3]=data_offset;
          data_offset+=Pc.Dim()*sizeof(real_t);
        }
      }
    }
    char* indx_ptr=&comp_data[0];
    Matrix<size_t> offset_indx(mat_cnt,1+(2+2)*(l1-l0), (size_t*)indx_ptr, false);
    for(size_t j=0;j<mat_cnt;j++){
      Matrix     <real_t>& M = mat[type][j];
      if(M.Dim(0)>0 && M.Dim(1)>0)
        memcpy(&comp_data[0]+offset_indx[j][0], &M[0][0], M.Dim(0)*M.Dim(1)*sizeof(real_t));
      for(size_t l=l0;l<l1;l++){
        Permutation<real_t>& Pr=getPerm_R(l,type,j);
        Permutation<real_t>& Pc=getPerm_C(l,type,j);
        if(Pr.Dim()>0){
          memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+0], &Pr.perm[0], Pr.Dim()*sizeof(size_t));
          memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+1], &Pr.scal[0], Pr.Dim()*sizeof(real_t));
        }
        if(Pc.Dim()>0){
          memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+2], &Pc.perm[0], Pc.Dim()*sizeof(size_t));
          memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+3], &Pc.scal[0], Pc.Dim()*sizeof(real_t));
        }
      }
    }
    return indx_size+mem_size;
  }

  void Save2File(const char* fname, bool replace=false){
    FILE* f = fopen(fname,"rb");
    if(f!=NULL) {
      fclose(f);
      if(!replace) return;
    }
    f = fopen(fname,"wb");
    if(f==NULL) return;
    int typeSize = sizeof(real_t);
    fwrite(&typeSize, sizeof(int), 1, f);    // save real_t's size for loading

    for (int i=0; i<PrecomputationType; i++) {
      int numRelCoords = mat[i].size();
      fwrite(&numRelCoords, sizeof(int), 1, f);
      for (int j=0; j<numRelCoords; j++) {
	Matrix<real_t>& M = mat[i][j];
	int dim1 = M.Dim(0), dim2 = M.Dim(1);
	fwrite(&dim1, sizeof(int), 1, f);
	fwrite(&dim2, sizeof(int), 1, f);
	if (dim1*dim2)
          fwrite(&M[0][0], sizeof(real_t), dim1*dim2, f);
      }
    }
    fclose(f);
  }

#define MY_FREAD(a,b,c,d) {			\
    size_t r_cnt=fread(a,b,c,d);		\
    if(r_cnt!=c){				\
      fputs ("Reading error ",stderr);		\
      exit (-1);				\
    } }

  void LoadFile(const char* fname){
    FILE* f = fopen(fname, "rb");
    size_t f_size = 0;
    char* f_data = NULL;

    if (f == NULL)
      std::cout << "No existing precomputation matrix file" << std::endl;
    else {
      struct stat fileStat;
      if (stat(fname, &fileStat) == 0) f_size = fileStat.st_size;
    }

    if (!f_size) return;
    else {
      f_data = new char [f_size];
      fseek(f, 0, SEEK_SET);
      MY_FREAD(f_data, sizeof(char), f_size, f);
      fclose(f);
    }

    char* f_ptr = f_data;
    int typeSize = *(int*)f_ptr; f_ptr += sizeof(int);
    assert(typeSize == sizeof(real_t));
    for(int i=0; i<PrecomputationType; i++){
      int numRelCoords = *(int*)f_ptr; f_ptr += sizeof(int);
      for(int j=0; j<numRelCoords; j++){
        Matrix<real_t>& M = mat[i][j];
        int dim1 = *(int*)f_ptr; f_ptr += sizeof(int);
        int dim2 = *(int*)f_ptr; f_ptr += sizeof(int);       
        if (dim1*dim2) {
          M.Resize(dim1, dim2);
          memcpy(&M[0][0], f_ptr, sizeof(real_t)*dim1*dim2);
          f_ptr += sizeof(real_t) * dim1 * dim2;
        }
      }
    }
    delete[] f_data;
  }

#undef MY_FREAD
};

}//end namespace

#endif //_PrecompMAT_HPP_
