#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_
#include "intrinsics.h"
#include "pvfmm.h"
#if FLOAT
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#else
typedef fftw_complex fft_complex;
typedef fftw_plan fft_plan;
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#endif
#include "geometry.h"

namespace pvfmm{
  struct PackedData{
    size_t len;
    Matrix<real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<real_t> scal[4*MAX_DEPTH];
    Matrix<real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData pt_interac_data;
  };
  struct VListData {
    size_t buff_size;
    size_t m;
    size_t n_blk0;
    std::vector<real_t*> precomp_mat;
    std::vector<std::vector<size_t> > fft_vec;
    std::vector<std::vector<size_t> > ifft_vec;
    std::vector<std::vector<real_t> > fft_scl;
    std::vector<std::vector<real_t> > ifft_scl;
    std::vector<std::vector<size_t> > interac_vec;
    std::vector<std::vector<size_t> > interac_dsp;
  };
  struct SetupData {
    int level;
    const Kernel* kernel;
    std::vector<Mat_Type> interac_type;
    std::vector<FMM_Node*> nodes_in ;
    std::vector<FMM_Node*> nodes_out;
    std::vector<Vector<real_t>*> input_vector;
    std::vector<Vector<real_t>*> output_vector;

    size_t M_dim0;
    size_t M_dim1;
    std::vector<size_t> interac_blk;
    std::vector<size_t> interac_cnt;
    std::vector<size_t> interac_mat;
    std::vector<size_t>  input_perm;
    std::vector<size_t> output_perm;

    ptSetupData pt_setup_data;
    VListData vlist_data;
    std::vector<char>* precomp_data;
    Matrix<real_t>* coord_data;
    Matrix<real_t>* input_data;
    Matrix<real_t>* output_data;
  };

class FMM_Tree {

private:
  int multipole_order;
  FMM_Node* root_node;
  std::vector<FMM_Node*> node_lst;
  const Kernel* kernel;
  PrecompMat* mat;
  Vector<char> dev_buffer;

  std::vector<Matrix<real_t> > node_data_buff;
  InteracList interacList;
  std::vector<std::vector<char> > precomp_lst;
  std::vector<SetupData > setup_data;
  std::vector<Vector<real_t> > upwd_check_surf;
  std::vector<Vector<real_t> > upwd_equiv_surf;
  std::vector<Vector<real_t> > dnwd_check_surf;
  std::vector<Vector<real_t> > dnwd_equiv_surf;

  fft_plan vprecomp_fftplan;
  bool vprecomp_fft_flag;
  fft_plan vlist_fftplan;
  bool vlist_fft_flag;
  fft_plan vlist_ifftplan;
  bool vlist_ifft_flag;

  Permutation<real_t> equiv_surf_perm(size_t m, size_t p_indx, const Permutation<real_t>& ker_perm, const Vector<real_t>* scal_exp=NULL){
    real_t eps=1e-10;
    int dof=ker_perm.Dim();

    real_t c[3]={-0.5,-0.5,-0.5};
    std::vector<real_t> trg_coord=d_check_surf(m,c,0);
    int n_trg=trg_coord.size()/3;

    Permutation<real_t> P=Permutation<real_t>(n_trg*dof);
    if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ) {
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(fabs(trg_coord[i*3+0]-trg_coord[j*3+0]*(p_indx==ReflecX?-1.0:1.0))<eps)
        if(fabs(trg_coord[i*3+1]-trg_coord[j*3+1]*(p_indx==ReflecY?-1.0:1.0))<eps)
        if(fabs(trg_coord[i*3+2]-trg_coord[j*3+2]*(p_indx==ReflecZ?-1.0:1.0))<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else if(p_indx==SwapXY || p_indx==SwapXZ){
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(fabs(trg_coord[i*3+0]-trg_coord[j*3+(p_indx==SwapXY?1:2)])<eps)
        if(fabs(trg_coord[i*3+1]-trg_coord[j*3+(p_indx==SwapXY?0:1)])<eps)
        if(fabs(trg_coord[i*3+2]-trg_coord[j*3+(p_indx==SwapXY?2:0)])<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else{
      for(int j=0;j<n_trg;j++){
        for(int k=0;k<dof;k++){
          P.perm[j*dof+k]=j*dof+ker_perm.perm[k];
        }
      }
    }

    if(scal_exp && p_indx==Scaling) {
      assert(dof==scal_exp->Dim());
      Vector<real_t> scal(scal_exp->Dim());
      for(size_t i=0;i<scal.Dim();i++){
        scal[i]=powf(2.0,(*scal_exp)[i]);
      }
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=scal[i];
        }
      }
    }
    {
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=ker_perm.scal[i];
        }
      }
    }
    return P;
  }

  inline int p2oLocal(std::vector<MortonId> & nodes, std::vector<MortonId>& leaves,
		      unsigned int maxNumPts, unsigned int maxDepth, bool complete) {
    assert(maxDepth<=MAX_DEPTH);
    std::vector<MortonId> leaves_lst;
    unsigned int init_size=leaves.size();
    unsigned int num_pts=nodes.size();
    MortonId curr_node=leaves[0];
    MortonId last_node=leaves[init_size-1].NextId();
    MortonId next_node;
    unsigned int curr_pt=0;
    unsigned int next_pt=curr_pt+maxNumPts;
    while(next_pt <= num_pts){
      next_node = curr_node.NextId();
      while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ){
	curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	next_node = curr_node.NextId();
      }
      leaves_lst.push_back(curr_node);
      curr_node = next_node;
      unsigned int inc=maxNumPts;
      while(next_pt < num_pts && curr_node > nodes[next_pt]){
	inc=inc<<1;
	next_pt+=inc;
	if(next_pt > num_pts){
	  next_pt = num_pts;
	  break;
	}
      }
      curr_pt = std::lower_bound(&nodes[0]+curr_pt,&nodes[0]+next_pt,curr_node,std::less<MortonId>())-&nodes[0];
      if(curr_pt >= num_pts) break;
      next_pt = curr_pt + maxNumPts;
      if(next_pt > num_pts) next_pt = num_pts;
    }
    if(complete) {
      while(curr_node<last_node){
	while( curr_node.NextId() > last_node && curr_node.GetDepth() < maxDepth-1 )
	  curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
	leaves_lst.push_back(curr_node);
	curr_node = curr_node.NextId();
      }
    }
    leaves=leaves_lst;
    return 0;
  }

  inline int points2Octree(const std::vector<MortonId>& pt_mid, std::vector<MortonId>& nodes,
			   unsigned int maxDepth, unsigned int maxNumPts) {
    int myrank=0, np=1;
    Profile::Tic("SortMortonId", true, 10);
    std::vector<MortonId> pt_sorted;
    HyperQuickSort(pt_mid, pt_sorted);
    size_t pt_cnt=pt_sorted.size();
    Profile::Toc();

    Profile::Tic("p2o_local", false, 10);
    std::vector<MortonId> nodes_local(1); nodes_local[0]=MortonId();
    p2oLocal(pt_sorted, nodes_local, maxNumPts, maxDepth, myrank==np-1);
    Profile::Toc();

    Profile::Tic("RemoveDuplicates", true, 10);
    size_t node_cnt=nodes_local.size();
    MortonId first_node;
    MortonId  last_node=nodes_local[node_cnt-1];
    size_t i=0;
    std::vector<MortonId> node_lst;
    if(myrank){
      while(i<node_cnt && nodes_local[i].getDFD(maxDepth)<first_node) i++; assert(i);
      last_node=nodes_local[i>0?i-1:0].NextId();

      while(first_node<last_node){
        while(first_node.isAncestor(last_node))
          first_node=first_node.getDFD(first_node.GetDepth()+1);
        if(first_node==last_node) break;
        node_lst.push_back(first_node);
        first_node=first_node.NextId();
      }
    }
    for(;i<node_cnt-(myrank==np-1?0:1);i++) node_lst.push_back(nodes_local[i]);
    nodes=node_lst;
    Profile::Toc();
    return 0;
  }

  void VListHadamard(size_t M_dim, std::vector<size_t>& interac_dsp,
                     std::vector<size_t>& interac_vec, std::vector<real_t*>& precomp_mat, Vector<real_t>& fft_in, Vector<real_t>& fft_out){
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =M_dim*chld_cnt*2;
    size_t fftsize_out=M_dim*chld_cnt*2;
    int err;
    real_t * zero_vec0, * zero_vec1;
    err = posix_memalign((void**)&zero_vec0, MEM_ALIGN, fftsize_in *sizeof(real_t));
    err = posix_memalign((void**)&zero_vec1, MEM_ALIGN, fftsize_out*sizeof(real_t));
    size_t n_out=fft_out.Dim()/fftsize_out;
#pragma omp parallel for
    for(size_t k=0;k<n_out;k++){
      Vector<real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
      dnward_check_fft.SetZero();
    }
    size_t mat_cnt=precomp_mat.size();
    size_t blk1_cnt=interac_dsp.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 4 / sizeof(real_t);
    real_t **IN_, **OUT_;
    err = posix_memalign((void**)&IN_ , MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    err = posix_memalign((void**)&OUT_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
#pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++){
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0;j<interac_cnt;j++){
        IN_ [BLOCK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[BLOCK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
      OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
    }
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++){
      size_t a=( pid   *M_dim)/omp_p;
      size_t b=((pid+1)*M_dim)/omp_p;
      for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
      for(size_t        k=a;        k<       b;       k++)
      for(size_t mat_indx=0; mat_indx< mat_cnt;mat_indx++){
        size_t interac_blk1 = blk1*mat_cnt+mat_indx;
        size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
        size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
        size_t interac_cnt  = interac_dsp1-interac_dsp0;
        real_t** IN = IN_ + BLOCK_SIZE*interac_blk1;
        real_t** OUT= OUT_+ BLOCK_SIZE*interac_blk1;
        real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2;
        for(size_t j=0;j<interac_cnt;j+=2){
          real_t* M_   = M;
          real_t* IN0  = IN [j+0] + k*chld_cnt*2;
          real_t* IN1  = IN [j+1] + k*chld_cnt*2;
          real_t* OUT0 = OUT[j+0] + k*chld_cnt*2;
          real_t* OUT1 = OUT[j+1] + k*chld_cnt*2;
#ifdef __SSE__
          if (j+2 < interac_cnt) {
            _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
          }
#endif
          matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
        }
      }
    }
    Profile::Add_FLOP(8*8*8*(interac_vec.size()/2)*M_dim);
    free(IN_ );
    free(OUT_);
    free(zero_vec0);
    free(zero_vec1);
  }

  template<typename ElemType>
  void CopyVec(std::vector<std::vector<ElemType> >& vec_, pvfmm::Vector<ElemType>& vec) {
    int omp_p=omp_get_max_threads();
    std::vector<size_t> vec_dsp(omp_p+1,0);
    for(size_t tid=0;tid<omp_p;tid++){
      vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
    }
    vec.Resize(vec_dsp[omp_p]);
#pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
    }
  }

  void PrecompAll(Mat_Type type, int level=-1) {
    if(level==-1) {
      for(int l=0;l<MAX_DEPTH;l++) {
        PrecompAll(type, l);
      }
      return;
    }
    for(size_t i=0;i<Perm_Count;i++) {
      PrecompPerm(type, (Perm_Type) i);
    }
    size_t mat_cnt=interacList.ListCount(type);
    mat->Mat(level, type, mat_cnt-1);
    std::vector<size_t> indx_lst;
    for(size_t i=0; i<mat_cnt; i++) {
      if(interacList.InteracClass(type,i)==i) {
        indx_lst.push_back(i);
      }
    }
    for(size_t i=0; i<indx_lst.size(); i++){
      Precomp(level, type, indx_lst[i]);
    }
    for(size_t mat_indx=0;mat_indx<mat_cnt;mat_indx++){
      Matrix<real_t>& M0=interacList.ClassMat(level, type, mat_indx);
      Permutation<real_t>& pr=interacList.Perm_R(level, type, mat_indx);
      Permutation<real_t>& pc=interacList.Perm_C(level, type, mat_indx);
      if(pr.Dim()!=M0.Dim(0) || pc.Dim()!=M0.Dim(1)) Precomp(level, type, mat_indx);
    }
  }

  Permutation<real_t>& PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<real_t>& P_ = mat->Perm(type, perm_indx);
    if(P_.Dim()!=0) return P_;
    size_t m=multipole_order;
    size_t p_indx=perm_indx % C_Perm;
    Permutation<real_t> P;
    switch (type) {
    case U2U_Type: {
      Vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm) {
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
      }else{
        ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
        scal_exp=kernel->k_m2m->src_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, &scal_exp);
      break;
    }
    case D2D_Type: {
      Vector<real_t> scal_exp;
      Permutation<real_t> ker_perm;
      if(perm_indx<C_Perm){
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
        for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
      }else{
        ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
        scal_exp=kernel->k_l2l->trg_scal;
      }
      P=equiv_surf_perm(m, p_indx, ker_perm, &scal_exp);
      break;
    }
    default:
      break;
    }
#pragma omp critical (PRECOMP_MATRIX_PTS)
    {
      if(P_.Dim()==0) P_=P;
    }
    return P_;
  }

  Matrix<real_t>& Precomp(int level, Mat_Type type, size_t mat_indx) {
    level=0;
    Matrix<real_t>& M_ = mat->Mat(level, type, mat_indx);
    if(M_.Dim(0)!=0 && M_.Dim(1)!=0) return M_;
    else{
      size_t class_indx = interacList.InteracClass(type, mat_indx);
      if(class_indx!=mat_indx){
        Matrix<real_t>& M0 = Precomp(level, type, class_indx);
        if(M0.Dim(0)==0 || M0.Dim(1)==0) return M_;

        for(size_t i=0;i<Perm_Count;i++) PrecompPerm(type, (Perm_Type) i);
        Permutation<real_t>& Pr = interacList.Perm_R(level, type, mat_indx);
        Permutation<real_t>& Pc = interacList.Perm_C(level, type, mat_indx);
        if(Pr.Dim()>0 && Pc.Dim()>0 && M0.Dim(0)>0 && M0.Dim(1)>0) return M_;
      }
    }
    Matrix<real_t> M;
    switch (type){
    case UC2UE0_Type:{
      if(!multipole_order) break;
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
      real_t eps=1, max_S=0;
      while(eps*(real_t)0.5+(real_t)1.0>1.0) eps*=0.5;
      for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
        if(fabs(S[i][i])>max_S) max_S=fabs(S[i][i]);
      }
      for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
      M=V.Transpose()*S;
      break;
    }
    case UC2UE1_Type:{
      if(!multipole_order) break;
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
      M=U.Transpose();
      break;
    }
    case DC2DE0_Type:{
      if(!multipole_order) break;
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
      real_t eps=1, max_S=0;
      while(eps*(real_t)0.5+(real_t)1.0>1.0) eps*=0.5;
      for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
        if(fabs(S[i][i])>max_S) max_S=fabs(S[i][i]);
      }
      for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
      M=V.Transpose()*S;
      break;
    }
    case DC2DE1_Type:{
      if(!multipole_order) break;
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
      M=U.Transpose();
      break;
    }
    case U2U_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2m->ker_dim;
      real_t c[3]={0,0,0};
      std::vector<real_t> check_surf=u_check_surf(multipole_order,c,level);
      size_t n_uc=check_surf.size()/3;
      real_t s=powf(0.5,(level+2));
      int* coord=interacList.RelativeCoord(type,mat_indx);
      real_t child_coord[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<real_t> equiv_surf=u_equiv_surf(multipole_order,child_coord,level+1);
      size_t n_ue=equiv_surf.size()/3;
      Matrix<real_t> M_ce2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
      kernel->k_m2m->BuildMatrix(&equiv_surf[0], n_ue,
                                 &check_surf[0], n_uc, &(M_ce2c[0][0]));
      Matrix<real_t>& M_c2e0 = Precomp(level, UC2UE0_Type, 0);
      Matrix<real_t>& M_c2e1 = Precomp(level, UC2UE1_Type, 0);
      M=(M_ce2c*M_c2e0)*M_c2e1;
      break;
    }
    case D2D_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_l2l->ker_dim;
      real_t s=powf(0.5,level+1);
      int* coord=interacList.RelativeCoord(type,mat_indx);
      real_t c[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
      std::vector<real_t> check_surf=d_check_surf(multipole_order,c,level);
      size_t n_dc=check_surf.size()/3;
      real_t parent_coord[3]={0,0,0};
      std::vector<real_t> equiv_surf=d_equiv_surf(multipole_order,parent_coord,level-1);
      size_t n_de=equiv_surf.size()/3;
      Matrix<real_t> M_pe2c(n_de*ker_dim[0],n_dc*ker_dim[1]);
      kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_de, &check_surf[0], n_dc, &(M_pe2c[0][0]));
      Matrix<real_t> M_c2e0=Precomp(level-1,DC2DE0_Type,0);
      Matrix<real_t> M_c2e1=Precomp(level-1,DC2DE1_Type,0);
      Permutation<real_t> ker_perm=kernel->k_l2l->perm_vec[C_Perm+Scaling];
      Vector<real_t> scal_exp=kernel->k_l2l->trg_scal;
      Permutation<real_t> P=equiv_surf_perm(multipole_order, Scaling, ker_perm, &scal_exp);
      M_c2e0=P*M_c2e0;
      ker_perm=kernel->k_l2l->perm_vec[0     +Scaling];
      scal_exp=kernel->k_l2l->src_scal;
      P=equiv_surf_perm(multipole_order, Scaling, ker_perm, &scal_exp);
      M_c2e1=M_c2e1*P;
      M=M_c2e0*(M_c2e1*M_pe2c);
      break;
    }
    case D2T_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_l2t->ker_dim;
      std::vector<real_t>& rel_trg_coord=mat->RelativeTrgCoord();
      real_t r=powf(0.5,level);
      size_t n_trg=rel_trg_coord.size()/3;
      std::vector<real_t> trg_coord(n_trg*3);
      for(size_t i=0;i<n_trg*3;i++) trg_coord[i]=rel_trg_coord[i]*r;
      real_t c[3]={0,0,0};
      std::vector<real_t> equiv_surf=d_equiv_surf(multipole_order,c,level);
      size_t n_eq=equiv_surf.size()/3;
      {
        M     .Resize(n_eq*ker_dim [0], n_trg*ker_dim [1]);
        kernel->k_l2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
      }
      Matrix<real_t>& M_c2e0=Precomp(level,DC2DE0_Type,0);
      Matrix<real_t>& M_c2e1=Precomp(level,DC2DE1_Type,0);
      M=M_c2e0*(M_c2e1*M);
      break;
    }
    case V_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      int n1=multipole_order*2;
      int n3 =n1*n1*n1;
      int n3_=n1*n1*(n1/2+1);
      real_t s=powf(0.5,level);
      int* coord2=interacList.RelativeCoord(type,mat_indx);
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
        if (!vprecomp_fft_flag){
          vprecomp_fftplan = fft_plan_many_dft_r2c(3, nnn, ker_dim[0]*ker_dim[1],
                                                   (real_t*)fftw_in, NULL, 1, n3,
                                                   (fft_complex*) fftw_out, NULL, 1, n3_,
                                                   FFTW_ESTIMATE);
          vprecomp_fft_flag=true;
        }
      }
      memcpy(fftw_in, &conv_poten[0], n3*ker_dim[0]*ker_dim[1]*sizeof(real_t));
      fft_execute_dft_r2c(vprecomp_fftplan, (real_t*)fftw_in, (fft_complex*)(fftw_out));
      Matrix<real_t> M_(2*n3_*ker_dim[0]*ker_dim[1],1,(real_t*)fftw_out,false);
      M=M_;
      free(fftw_in);
      free(fftw_out);
      break;
    }
    case V1_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2l->ker_dim;
      size_t mat_cnt =interacList.ListCount( V_Type);
      for(size_t k=0;k<mat_cnt;k++) Precomp(level, V_Type, k);

      const size_t chld_cnt=1UL<<3;
      size_t n1=multipole_order*2;
      size_t M_dim=n1*n1*(n1/2+1);
      size_t n3=n1*n1*n1;

      Vector<real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2);
      zero_vec.SetZero();

      Vector<real_t*> M_ptr(chld_cnt*chld_cnt);
      for(size_t i=0;i<chld_cnt*chld_cnt;i++) M_ptr[i]=&zero_vec[0];

      int* rel_coord_=interacList.RelativeCoord(V1_Type, mat_indx);
      for(int j1=0;j1<chld_cnt;j1++)
        for(int j2=0;j2<chld_cnt;j2++){
          int rel_coord[3]={rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                            rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                            rel_coord_[2]*2-(j1/4)%2+(j2/4)%2};
          for(size_t k=0;k<mat_cnt;k++){
            int* ref_coord=interacList.RelativeCoord(V_Type, k);
            if(ref_coord[0]==rel_coord[0] &&
               ref_coord[1]==rel_coord[1] &&
               ref_coord[2]==rel_coord[2]){
              Matrix<real_t>& M = mat->Mat(level, V_Type, k);
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
    case W_Type:{
      if(!multipole_order) break;
      const int* ker_dim=kernel->k_m2t->ker_dim;
      std::vector<real_t>& rel_trg_coord=mat->RelativeTrgCoord();
      real_t s=powf(0.5,level);
      size_t n_trg=rel_trg_coord.size()/3;
      std::vector<real_t> trg_coord(n_trg*3);
      for(size_t j=0;j<n_trg*3;j++) trg_coord[j]=rel_trg_coord[j]*s;
      int* coord2=interacList.RelativeCoord(type,mat_indx);
      real_t c[3]={(real_t)((coord2[0]+1)*s*0.25),(real_t)((coord2[1]+1)*s*0.25),(real_t)((coord2[2]+1)*s*0.25)};
      std::vector<real_t> equiv_surf=u_equiv_surf(multipole_order,c,level+1);
      size_t n_eq=equiv_surf.size()/3;
      {
        M     .Resize(n_eq*ker_dim [0],n_trg*ker_dim [1]);
        kernel->k_m2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
      }
      break;
    }
    case BC_Type:{
      if(!multipole_order) break;
      if(kernel->k_m2l->ker_dim[0]!=kernel->k_m2m->ker_dim[0]) break;
      if(kernel->k_m2l->ker_dim[1]!=kernel->k_l2l->ker_dim[1]) break;
      int ker_dim[2]={kernel->k_m2l->ker_dim[0],kernel->k_m2l->ker_dim[1]};
      size_t mat_cnt_m2m=interacList.ListCount(U2U_Type);
      size_t n_surf=(6*(multipole_order-1)*(multipole_order-1)+2);
      if((M.Dim(0)!=n_surf*ker_dim[0] || M.Dim(1)!=n_surf*ker_dim[1]) && level==0){
        Matrix<real_t> M_m2m[MAX_DEPTH+1];
        Matrix<real_t> M_m2l[MAX_DEPTH+1];
        Matrix<real_t> M_l2l[MAX_DEPTH+1];
        Matrix<real_t> M_equiv_zero_avg(n_surf*ker_dim[0],n_surf*ker_dim[0]);
        Matrix<real_t> M_check_zero_avg(n_surf*ker_dim[1],n_surf*ker_dim[1]);
        Matrix<real_t> M_s2c;
        int ker_dim[2]={kernel->k_m2m->ker_dim[0],kernel->k_m2m->ker_dim[1]};
        M_s2c.ReInit(ker_dim[0],n_surf*ker_dim[1]);
        std::vector<real_t> uc_coord;
        real_t c[3]={0,0,0};
        uc_coord=u_check_surf(multipole_order,c,0);
#pragma omp parallel for schedule(dynamic)
        for(size_t i=0;i<n_surf;i++){
          std::vector<real_t> M_=cheb_integ(0, &uc_coord[i*3], 1.0, *kernel->k_m2m);
          for(size_t j=0; j<ker_dim[0]; j++)
            for(int k=0; k<ker_dim[1]; k++)
              M_s2c[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
        }
        Matrix<real_t>& M_c2e0 = Precomp(level, UC2UE0_Type, 0);
        Matrix<real_t>& M_c2e1 = Precomp(level, UC2UE1_Type, 0);
        Matrix<real_t> M_s2e=(M_s2c*M_c2e0)*M_c2e1;
        for(size_t i=0;i<M_s2e.Dim(0);i++) {
          real_t s=0;
          for(size_t j=0;j<M_s2e.Dim(1);j++) s+=M_s2e[i][j];
          s=1.0/s;
          for(size_t j=0;j<M_s2e.Dim(1);j++) M_s2e[i][j]*=s;
        }

        assert(M_equiv_zero_avg.Dim(0)==M_s2e.Dim(1));
        assert(M_equiv_zero_avg.Dim(1)==M_s2e.Dim(1));
        M_equiv_zero_avg.SetZero();
        for(size_t i=0;i<n_surf*ker_dim[0];i++)
          M_equiv_zero_avg[i][i]=1;
        for(size_t i=0;i<n_surf;i++)
          for(size_t k=0;k<ker_dim[0];k++)
            for(size_t j=0;j<n_surf*ker_dim[0];j++)
              M_equiv_zero_avg[i*ker_dim[0]+k][j]-=M_s2e[k][j];
        M_check_zero_avg.SetZero();
        for(size_t i=0;i<n_surf*ker_dim[1];i++)
          M_check_zero_avg[i][i]+=1;
        for(size_t i=0;i<n_surf;i++)
          for(size_t j=0;j<n_surf;j++)
            for(size_t k=0;k<ker_dim[1];k++)
              M_check_zero_avg[i*ker_dim[1]+k][j*ker_dim[1]+k]-=1.0/n_surf;
        for(int level=0; level>=-MAX_DEPTH; level--){
          Precomp(level, D2D_Type, 0);
          Permutation<real_t>& Pr = interacList.Perm_R(level, D2D_Type, 0);
          Permutation<real_t>& Pc = interacList.Perm_C(level, D2D_Type, 0);
          M_l2l[-level] = M_check_zero_avg * Pr * Precomp(level, D2D_Type, interacList.InteracClass(D2D_Type, 0)) * Pc * M_check_zero_avg;
          assert(M_l2l[-level].Dim(0)>0 && M_l2l[-level].Dim(1)>0);
          for(size_t mat_indx=0; mat_indx<mat_cnt_m2m; mat_indx++){
            Precomp(level, U2U_Type, mat_indx);
            Permutation<real_t>& Pr = interacList.Perm_R(level, U2U_Type, mat_indx);
            Permutation<real_t>& Pc = interacList.Perm_C(level, U2U_Type, mat_indx);
            Matrix<real_t> M = Pr * Precomp(level, U2U_Type, interacList.InteracClass(U2U_Type, mat_indx)) * Pc;
            assert(M.Dim(0)>0 && M.Dim(1)>0);

            if(mat_indx==0) M_m2m[-level] = M_equiv_zero_avg*M*M_equiv_zero_avg;
            else M_m2m[-level] += M_equiv_zero_avg*M*M_equiv_zero_avg;
          }
          if(level==0){
            real_t s=(1UL<<(-level));
            real_t dc_coord[3]={0,0,0};
            std::vector<real_t> trg_coord=d_check_surf(multipole_order, dc_coord, level);
            Matrix<real_t> M_ue2dc(n_surf*ker_dim[0], n_surf*ker_dim[1]); M_ue2dc.SetZero();

            for(int x0=-2;x0<4;x0++)
              for(int x1=-2;x1<4;x1++)
                for(int x2=-2;x2<4;x2++)
                  if(abs(x0)>1 || abs(x1)>1 || abs(x2)>1){
                    real_t ue_coord[3]={x0*s, x1*s, x2*s};
                    std::vector<real_t> src_coord=u_equiv_surf(multipole_order, ue_coord, level);
                    Matrix<real_t> M_tmp(n_surf*ker_dim[0], n_surf*ker_dim[1]);
                    kernel->k_m2l->BuildMatrix(&src_coord[0], n_surf, &trg_coord[0], n_surf, &(M_tmp[0][0]));
                    M_ue2dc+=M_tmp;
                  }
            M_m2l[-level]=M_equiv_zero_avg*M_ue2dc * M_check_zero_avg;
          }else{
            M_m2l[-level]=M_equiv_zero_avg * M_m2l[-level-1] * M_check_zero_avg;
            Permutation<real_t> ker_perm=kernel->k_m2l->perm_vec[0     +Scaling];
            Vector<real_t> scal_exp=kernel->k_m2l->src_scal;
            for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
            Permutation<real_t> P=equiv_surf_perm(multipole_order, Scaling, ker_perm, &scal_exp);
            M_m2l[-level]=P*M_m2l[-level];
            ker_perm=kernel->k_m2l->perm_vec[C_Perm+Scaling];
            scal_exp=kernel->k_m2l->trg_scal;
            for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
            P=equiv_surf_perm(multipole_order, Scaling, ker_perm, &scal_exp);
            M_m2l[-level]=M_m2l[-level]*P;
          }
        }
        for(int level=-MAX_DEPTH;level<=0;level++){
          if(level==-MAX_DEPTH) M = M_m2l[-level];
          else                  M = M_equiv_zero_avg * (M_m2l[-level] + M_m2m[-level]*M*M_l2l[-level]) * M_check_zero_avg;
        }
        std::vector<real_t> corner_pts;
        corner_pts.push_back(0); corner_pts.push_back(0); corner_pts.push_back(0);
        corner_pts.push_back(1); corner_pts.push_back(0); corner_pts.push_back(0);
        corner_pts.push_back(0); corner_pts.push_back(1); corner_pts.push_back(0);
        corner_pts.push_back(0); corner_pts.push_back(0); corner_pts.push_back(1);
        corner_pts.push_back(0); corner_pts.push_back(1); corner_pts.push_back(1);
        corner_pts.push_back(1); corner_pts.push_back(0); corner_pts.push_back(1);
        corner_pts.push_back(1); corner_pts.push_back(1); corner_pts.push_back(0);
        corner_pts.push_back(1); corner_pts.push_back(1); corner_pts.push_back(1);
        size_t n_corner=corner_pts.size()/3;
        c[0]=0; c[1]=0; c[2]=0;
        std::vector<real_t> up_equiv_surf=u_equiv_surf(multipole_order,c,0);
        std::vector<real_t> dn_equiv_surf=d_equiv_surf(multipole_order,c,0);
        std::vector<real_t> dn_check_surf=d_check_surf(multipole_order,c,0);

        Matrix<real_t> M_err;
        Matrix<real_t> M_e2pt(n_surf*kernel->k_l2l->ker_dim[0],n_corner*kernel->k_l2l->ker_dim[1]);
        kernel->k_l2l->BuildMatrix(&dn_equiv_surf[0], n_surf,
                                   &corner_pts[0], n_corner, &(M_e2pt[0][0]));
        Matrix<real_t>& M_dc2de0 = Precomp(0, DC2DE0_Type, 0);
        Matrix<real_t>& M_dc2de1 = Precomp(0, DC2DE1_Type, 0);
        M_err=(M*M_dc2de0)*(M_dc2de1*M_e2pt);
        for(size_t k=0;k<n_corner;k++) {
          for(int j0=-1;j0<=1;j0++)
            for(int j1=-1;j1<=1;j1++)
              for(int j2=-1;j2<=1;j2++){
                real_t pt_c[3]={corner_pts[k*3+0]-j0,
                                corner_pts[k*3+1]-j1,
                                corner_pts[k*3+2]-j2};
                if(fabs(pt_c[0]-0.5)>1.0 || fabs(pt_c[1]-0.5)>1.0 || fabs(pt_c[2]-0.5)>1.0){
                  Matrix<real_t> M_e2pt(n_surf*ker_dim[0],ker_dim[1]);
                  kernel->k_m2l->BuildMatrix(&up_equiv_surf[0], n_surf,
                                             &pt_c[0], 1, &(M_e2pt[0][0]));
                  for(size_t i=0;i<M_e2pt.Dim(0);i++)
                    for(size_t j=0;j<M_e2pt.Dim(1);j++)
                      M_err[i][k*ker_dim[1]+j]+=M_e2pt[i][j];
                }
              }
        }
        Matrix<real_t> M_grad(M_err.Dim(0),n_surf*ker_dim[1]);
        for(size_t i=0;i<M_err.Dim(0);i++)
          for(size_t k=0;k<ker_dim[1];k++)
            for(size_t j=0;j<n_surf;j++){
              M_grad[i][j*ker_dim[1]+k]=  M_err[i][0*ker_dim[1]+k]
                +(M_err[i][1*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*3+0]
                +(M_err[i][2*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*3+1]
                +(M_err[i][3*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*3+2]
                +(M_err[i][4*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][2*ker_dim[1]+k]-M_err[i][3*ker_dim[1]+k])*dn_check_surf[j*3+1]*dn_check_surf[j*3+2]
                +(M_err[i][5*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][1*ker_dim[1]+k]-M_err[i][3*ker_dim[1]+k])*dn_check_surf[j*3+2]*dn_check_surf[j*3+0]
                +(M_err[i][6*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][1*ker_dim[1]+k]-M_err[i][2*ker_dim[1]+k])*dn_check_surf[j*3+0]*dn_check_surf[j*3+1]
                +(M_err[i][7*ker_dim[1]+k]+M_err[i][1*ker_dim[1]+k]+M_err[i][2*ker_dim[1]+k]+M_err[i][3*ker_dim[1]+k]
                  -M_err[i][0*ker_dim[1]+k]-M_err[i][4*ker_dim[1]+k]-M_err[i][5*ker_dim[1]+k]-M_err[i][6*ker_dim[1]+k])*dn_check_surf[j*3+0]*dn_check_surf[j*3+1]*dn_check_surf[j*3+2];
            }
        M-=M_grad;
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

public:

  FMM_Tree(): root_node(NULL), vprecomp_fft_flag(false), vlist_fft_flag(false),
	      vlist_ifft_flag(false), mat(NULL), kernel(NULL) { };

  ~FMM_Tree(){
    if(root_node!=NULL){
      delete root_node;
    }
    if(mat!=NULL){
      delete mat;
      mat=NULL;
    }
    if(vprecomp_fft_flag) fft_destroy_plan(vprecomp_fftplan);
    {
      if(vlist_fft_flag ) fft_destroy_plan(vlist_fftplan );
      if(vlist_ifft_flag) fft_destroy_plan(vlist_ifftplan);
      vlist_fft_flag =false;
      vlist_ifft_flag=false;
    }

  }

  void Initialize(InitData* init_data) {
    Profile::Tic("InitTree",true);{
      Profile::Tic("InitRoot",false,5);
      int max_depth=init_data->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
      if(root_node) delete root_node;
      root_node=new FMM_Node();
      root_node->Initialize(NULL,0,init_data);
      Profile::Toc();

      Profile::Tic("Points2Octee",true,5);
      std::vector<MortonId> lin_oct;
      std::vector<MortonId> pt_mid;
      Vector<real_t>& pt_c=root_node->pt_coord;
      size_t pt_cnt=pt_c.Dim()/3;
      pt_mid.resize(pt_cnt);
#pragma omp parallel for
      for(size_t i=0;i<pt_cnt;i++){
        pt_mid[i]=MortonId(pt_c[i*3+0],pt_c[i*3+1],pt_c[i*3+2],max_depth);
      }
      points2Octree(pt_mid,lin_oct,max_depth,init_data->max_pts);
      Profile::Toc();

      Profile::Tic("ScatterPoints",true,5);
      std::vector<Vector<real_t>*> coord_lst;
      std::vector<Vector<real_t>*> value_lst;
      std::vector<Vector<size_t>*> scatter_lst;
      root_node->NodeDataVec(coord_lst, value_lst, scatter_lst);
      assert(coord_lst.size()==value_lst.size());

      std::vector<size_t> scatter_index;
      for(size_t i=0;i<coord_lst.size();i++){
        if(!coord_lst[i]) continue;
        Vector<real_t>& pt_c=*coord_lst[i];
        size_t pt_cnt=pt_c.Dim()/3;
        pt_mid.resize(pt_cnt);
#pragma omp parallel for
        for(size_t i=0;i<pt_cnt;i++){
    	  pt_mid[i]=MortonId(pt_c[i*3+0],pt_c[i*3+1],pt_c[i*3+2],max_depth);
        }
        SortIndex(pt_mid, scatter_index, &lin_oct[0]);
        Forward  (pt_c, scatter_index);
        if(value_lst[i]!=NULL){
          Vector<real_t>& pt_v=*value_lst[i];
          Forward(pt_v, scatter_index);
        }
        if(scatter_lst[i]!=NULL){
          Vector<size_t>& pt_s=*scatter_lst[i];
          memcpy(pt_s.data_ptr, &scatter_index[0], pt_s.Dim()*sizeof(size_t));
        }
      }
      Profile::Toc();

      Profile::Tic("PointerTree",false,5);
      int omp_p=omp_get_max_threads();
      root_node->SetGhost(false);
      for(int i=0;i<omp_p;i++){
        size_t idx=(lin_oct.size()*i)/omp_p;
        FMM_Node* n=FindNode(lin_oct[idx], true);
        assert(n->GetMortonId()==lin_oct[idx]);
      }
#pragma omp parallel for
      for(int i=0;i<omp_p;i++){
        size_t a=(lin_oct.size()* i   )/omp_p;
        size_t b=(lin_oct.size()*(i+1))/omp_p;
        size_t idx=a;
        FMM_Node* n=FindNode(lin_oct[idx], false);
        if(a==0) n=root_node;
        while(n!=NULL && (idx<b || i==omp_p-1)){
          n->SetGhost(false);
          MortonId dn=n->GetMortonId();
          if(idx<b && dn.isAncestor(lin_oct[idx])){
            if(n->IsLeaf()) n->Subdivide();
          }else if(idx<b && dn==lin_oct[idx]){
            if(!n->IsLeaf()) n->Truncate();
            assert(n->IsLeaf());
            idx++;
          }else{
            n->Truncate();
            n->SetGhost(true);
          }
          n=PreorderNxt(n);
        }
      }
      Profile::Toc();
      Profile::Tic("InitFMMData",true,5);
      std::vector<FMM_Node*>& nodes=GetNodeList();
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        if(nodes[i]->FMMData()==NULL) nodes[i]->FMMData()=new FMM_Data();
      }
      Profile::Toc();
    }
    Profile::Toc();
  }

  void Initialize(int mult_order, const Kernel* kernel_) {
    Profile::Tic("InitFMM_Pts",true);{
    int rank=0;
    bool verbose=false;
    if(kernel_) kernel_->Initialize(verbose);
    multipole_order=mult_order;
    kernel=kernel_;
    assert(kernel!=NULL);
    bool save_precomp=false;
    mat=new PrecompMat(true);
    std::string mat_fname;
    if(mat_fname.size()==0){
      std::stringstream st;
      if(!st.str().size()){
        char* pvfmm_dir = getenv ("PVFMM_DIR");
        if(pvfmm_dir) st<<pvfmm_dir;
      }
#ifndef STAT_MACROS_BROKEN
      if(st.str().size()){
        struct stat stat_buff;
        if(stat(st.str().c_str(), &stat_buff) || !S_ISDIR(stat_buff.st_mode)){
          std::cout<<"error: path not found: "<<st.str()<<'\n';
          exit(0);
        }
      }
#endif
      if(st.str().size()) st<<'/';
      st<<"Precomp_"<<kernel->ker_name.c_str()<<"_m"<<mult_order;
      if(sizeof(real_t)==8) st<<"";
      else if(sizeof(real_t)==4) st<<"_f";
      else st<<"_t"<<sizeof(real_t);
      st<<".data";
      mat_fname=st.str();
      save_precomp=true;
    }
    mat->LoadFile(mat_fname.c_str());
    interacList.Initialize(mat);
    Profile::Tic("PrecompUC2UE",false,4);
    PrecompAll(UC2UE0_Type);
    PrecompAll(UC2UE1_Type);
    Profile::Toc();
    Profile::Tic("PrecompDC2DE",false,4);
    PrecompAll(DC2DE0_Type);
    PrecompAll(DC2DE1_Type);
    Profile::Toc();
    Profile::Tic("PrecompBC",false,4);
    PrecompAll(BC_Type,0);
    Profile::Toc();
    Profile::Tic("PrecompU2U",false,4);
    PrecompAll(U2U_Type);
    Profile::Toc();
    Profile::Tic("PrecompD2D",false,4);
    PrecompAll(D2D_Type);
    Profile::Toc();
    if(save_precomp){
      Profile::Tic("Save2File",false,4);
      if(!rank){
        FILE* f=fopen(mat_fname.c_str(),"r");
        if(f==NULL) { //File does not exists.
          mat->Save2File(mat_fname.c_str());
        }else fclose(f);
      }
      Profile::Toc();
    }
    Profile::Tic("PrecompV",false,4);
    PrecompAll(V_Type);
    Profile::Toc();
    Profile::Tic("PrecompV1",false,4);
    PrecompAll(V1_Type);
    Profile::Toc();
    }Profile::Toc();
  }

  FMM_Node* PreorderNxt(FMM_Node* curr_node) {
    assert(curr_node!=NULL);
    int n=(1UL<<3);
    if(!curr_node->IsLeaf())
      for(int i=0;i<n;i++)
	if(curr_node->Child(i)!=NULL)
	  return curr_node->Child(i);
    FMM_Node* node=curr_node;
    while(true){
      int i=node->Path2Node()+1;
      node=node->Parent();
      if(node==NULL) return NULL;
      for(;i<n;i++)
	if(node->Child(i)!=NULL)
	  return node->Child(i);
    }
  }

  void SetColleagues(FMM_Node* node=NULL) {
    int n1=27;
    int n2=8;
    if(node==NULL){
      FMM_Node* curr_node=root_node;
      if(curr_node!=NULL){
	curr_node->SetColleague(curr_node,(n1-1)/2);
        curr_node=PreorderNxt(curr_node);
      }
      std::vector<std::vector<FMM_Node*> > nodes(MAX_DEPTH);
      while(curr_node!=NULL){
        nodes[curr_node->depth].push_back(curr_node);
        curr_node=PreorderNxt(curr_node);
      }
      for(size_t i=0;i<MAX_DEPTH;i++){
        size_t j0=nodes[i].size();
        FMM_Node** nodes_=&nodes[i][0];
#pragma omp parallel for
        for(size_t j=0;j<j0;j++){
          SetColleagues(nodes_[j]);
        }
      }
    }else{
      FMM_Node* parent_node;
      FMM_Node* tmp_node1;
      FMM_Node* tmp_node2;
      for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
      parent_node=node->Parent();
      if(parent_node==NULL) return;
      int l=node->Path2Node();
      for(int i=0;i<n1;i++){
        tmp_node1=parent_node->Colleague(i);
        if(tmp_node1!=NULL)
        if(!tmp_node1->IsLeaf()){
          for(int j=0;j<n2;j++){
            tmp_node2=tmp_node1->Child(j);
            if(tmp_node2!=NULL){

              bool flag=true;
              int a=1,b=1,new_indx=0;
              for(int k=0;k<3;k++){
                int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
                if(-1>indx_diff || indx_diff>1) flag=false;
                new_indx+=(indx_diff+1)*b;
                a*=2;b*=3;
              }
              if(flag){
                node->SetColleague(tmp_node2,new_indx);
              }
            }
          }
        }
      }
    }
  }

  std::vector<FMM_Node*>& GetNodeList() {
    if(root_node->GetStatus() & 1){
      node_lst.clear();
      FMM_Node* n=root_node;
      while(n!=NULL){
	int& status=n->GetStatus();
	status=(status & (~(int)1));
	node_lst.push_back(n);
	n=PreorderNxt(n);
      }
    }
    return node_lst;
  }

  FMM_Node* FindNode(MortonId& key, bool subdiv, FMM_Node* start=NULL) {
    int num_child=1UL<<3;
    FMM_Node* n=start;
    if(n==NULL) n=root_node;
    while(n->GetMortonId()<key && (!n->IsLeaf()||subdiv)){
      if(n->IsLeaf() && !n->IsGhost()) n->Subdivide();
      if(n->IsLeaf()) break;
      for(int j=0;j<num_child;j++){
	if(n->Child(j)->GetMortonId().NextId()>key){
	  n=n->Child(j);
	  break;
	}
      }
    }
    assert(!subdiv || n->IsGhost() || n->GetMortonId()==key);
    return n;
  }

  FMM_Node* PostorderFirst() {
    FMM_Node* node=root_node;
    int n=(1UL<<3);
    while(true){
      if(node->IsLeaf()) return node;
      for(int i=0;i<n;i++) {
	if(node->Child(i)!=NULL){
	  node=node->Child(i);
	  break;
	}
      }
    }
  }

  FMM_Node* PostorderNxt(FMM_Node* curr_node) {
    assert(curr_node!=NULL);
    FMM_Node* node=curr_node;
    int j=node->Path2Node()+1;
    node=node->Parent();
    if(node==NULL) return NULL;
    int n=(1UL<<3);
    for(;j<n;j++){
      if(node->Child(j)!=NULL){
	node=node->Child(j);
	while(true){
	  if(node->IsLeaf()) return node;
	  for(int i=0;i<n;i++) {
	    if(node->Child(i)!=NULL){
	      node=node->Child(i);
	      break;
	    }
	  }
	}
      }
    }
    return node;
  }

  void InitFMM_Tree(bool refine) {
    Profile::Tic("InitFMM_Tree",true);{
      interacList.Initialize(mat);
    }Profile::Toc();
  }

  void SetupFMM() {
    Profile::Tic("SetupFMM",true);{
    Profile::Tic("SetColleagues",false,3);
    SetColleagues();
    Profile::Toc();
    Profile::Tic("CollectNodeData",false,3);
    FMM_Node* n=dynamic_cast<FMM_Node*>(PostorderFirst());
    std::vector<FMM_Node*> all_nodes;
    while(n!=NULL){
      n->pt_cnt[0]=0;
      n->pt_cnt[1]=0;
      all_nodes.push_back(n);
      n=static_cast<FMM_Node*>(PostorderNxt(n));
    }
    std::vector<std::vector<FMM_Node*> > node_lists; // TODO: Remove this parameter, not really needed
    CollectNodeData(all_nodes, node_data_buff, node_lists);
    Profile::Toc();

    Profile::Tic("BuildLists",false,3);
    BuildInteracLists();
    Profile::Toc();
    setup_data.resize(8*MAX_DEPTH);
    precomp_lst.resize(8);
    Profile::Tic("UListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*0].precomp_data=&precomp_lst[0];
      U_ListSetup(setup_data[i+MAX_DEPTH*0],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();
    Profile::Tic("WListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*1].precomp_data=&precomp_lst[1];
      W_ListSetup(setup_data[i+MAX_DEPTH*1],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();
    Profile::Tic("XListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*2].precomp_data=&precomp_lst[2];
      X_ListSetup(setup_data[i+MAX_DEPTH*2],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();
    Profile::Tic("VListSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*3].precomp_data=&precomp_lst[3];
      V_ListSetup(setup_data[i+MAX_DEPTH*3],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();
    Profile::Tic("D2DSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*4].precomp_data=&precomp_lst[4];
      Down2DownSetup(setup_data[i+MAX_DEPTH*4],node_data_buff,node_lists,i);
    }
    Profile::Toc();
    Profile::Tic("D2TSetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*5].precomp_data=&precomp_lst[5];
      Down2TargetSetup(setup_data[i+MAX_DEPTH*5],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();

    Profile::Tic("S2USetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*6].precomp_data=&precomp_lst[6];
      Source2UpSetup(setup_data[i+MAX_DEPTH*6],node_data_buff,node_lists,i==0?-1:MAX_DEPTH+1);
    }
    Profile::Toc();
    Profile::Tic("U2USetup",false,3);
    for(size_t i=0;i<MAX_DEPTH;i++){
      setup_data[i+MAX_DEPTH*7].precomp_data=&precomp_lst[7];
      Up2UpSetup(setup_data[i+MAX_DEPTH*7],node_data_buff,node_lists,i);
    }
    Profile::Toc();
    ClearFMMData();
    }Profile::Toc();
  }

  void ClearFMMData() {
    Profile::Tic("ClearFMMData",true);
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      Matrix<real_t>* mat;
      mat=setup_data[0+MAX_DEPTH*1]. input_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
      mat=setup_data[0+MAX_DEPTH*2].output_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
      mat=setup_data[0+MAX_DEPTH*0].output_data;
      if(mat && mat->Dim(0)*mat->Dim(1)){
        size_t a=(mat->Dim(0)*mat->Dim(1)*(j+0))/omp_p;
        size_t b=(mat->Dim(0)*mat->Dim(1)*(j+1))/omp_p;
        memset(&(*mat)[0][a],0,(b-a)*sizeof(real_t));
      }
    }
    Profile::Toc();
  }

  void CollectNodeData(std::vector<FMM_Node*>& node, std::vector<Matrix<real_t> >& buff_list, std::vector<std::vector<FMM_Node*> >& n_list) {
    std::vector<std::vector<Vector<real_t>* > > vec_list(0);
    if(buff_list.size()<7) buff_list.resize(7);
    if(   n_list.size()<7)    n_list.resize(7);
    if( vec_list.size()<7)  vec_list.resize(7);
    int omp_p=omp_get_max_threads();
    if(node.size()==0) return;
    int indx=0;
    size_t vec_sz;
    Matrix<real_t>& M_uc2ue = interacList.ClassMat(0, UC2UE1_Type, 0);
    vec_sz=M_uc2ue.Dim(1);
    std::vector< FMM_Node* > node_lst;
    node_lst.clear();
    std::vector<std::vector< FMM_Node* > > node_lst_vec(MAX_DEPTH+1);
    FMM_Node* r_node=NULL;
    for(size_t i=0;i<node.size();i++){
      if(!node[i]->IsLeaf()){
        node_lst_vec[node[i]->depth].push_back(node[i]);
      }else{
        node[i]->pt_cnt[0]+=node[i]-> src_coord.Dim()/3;
        node[i]->pt_cnt[0]+=node[i]->surf_coord.Dim()/3;
        if(node[i]->IsGhost()) node[i]->pt_cnt[0]++; // TODO: temporary fix, pt_cnt not known for ghost nodes
      }
      if(node[i]->depth==0) r_node=node[i];
    }
    size_t chld_cnt=1UL<<3;
    for(int i=MAX_DEPTH;i>=0;i--){
      for(size_t j=0;j<node_lst_vec[i].size();j++){
        for(size_t k=0;k<chld_cnt;k++){
          FMM_Node* node=node_lst_vec[i][j]->Child(k);
          node_lst_vec[i][j]->pt_cnt[0]+=node->pt_cnt[0];
        }
      }
    }
    for(int i=0;i<=MAX_DEPTH;i++){
      for(size_t j=0;j<node_lst_vec[i].size();j++){
        if(node_lst_vec[i][j]->pt_cnt[0])
          for(size_t k=0;k<chld_cnt;k++){
            FMM_Node* node=node_lst_vec[i][j]->Child(k);
            node_lst.push_back(node);
          }
      }
    }
    if(r_node!=NULL) node_lst.push_back(r_node);
    n_list[indx]=node_lst;
    std::vector<Vector<real_t>*>& vec_lst=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){
      FMM_Node* node=node_lst[i];
      Vector<real_t>& data_vec=node->FMMData()->upward_equiv;
//std::cout << "CollectNodeData upward: " << data_vec.Dim() << std::endl;
      data_vec.Resize(vec_sz);
      vec_lst.push_back(&data_vec);
    }

    indx=1;
    Matrix<real_t>& M_dc2de0 = interacList.ClassMat(0, DC2DE0_Type, 0);
    vec_sz=M_dc2de0.Dim(0);
    node_lst.clear();
    for(int i=0;i<=MAX_DEPTH;i++)
      node_lst_vec[i].clear();
    r_node=NULL;
    for(size_t i=0;i<node.size();i++){
      if(!node[i]->IsLeaf()){
        node_lst_vec[node[i]->depth].push_back(node[i]);
      }else{
        node[i]->pt_cnt[1]+=node[i]->trg_coord.Dim()/3;
      }
      if(node[i]->depth==0) r_node=node[i];
    }
    chld_cnt=1UL<<3;
    for(int i=MAX_DEPTH;i>=0;i--){
      for(size_t j=0;j<node_lst_vec[i].size();j++){
        for(size_t k=0;k<chld_cnt;k++){
          FMM_Node* node=node_lst_vec[i][j]->Child(k);
          node_lst_vec[i][j]->pt_cnt[1]+=node->pt_cnt[1];
        }
      }
    }
    for(int i=0;i<=MAX_DEPTH;i++){
      for(size_t j=0;j<node_lst_vec[i].size();j++){
        if(node_lst_vec[i][j]->pt_cnt[1])
          for(size_t k=0;k<chld_cnt;k++){
            FMM_Node* node=node_lst_vec[i][j]->Child(k);
            node_lst.push_back(node);
          }
      }
    }
    if(r_node!=NULL) node_lst.push_back(r_node);
    n_list[indx]=node_lst;
    std::vector<Vector<real_t>*>& vec_lst1=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){
      FMM_Node* node=node_lst[i];
      Vector<real_t>& data_vec=node->FMMData()->dnward_equiv;
//std::cout << "CollectNodeData downward: " << data_vec.Dim() << std::endl;
      data_vec.Resize(vec_sz);
      vec_lst1.push_back(&data_vec);
    }

    indx=2;
    node_lst.clear();
    for(int i=0;i<=MAX_DEPTH;i++)
      node_lst_vec[i].clear();
    for(size_t i=0;i<node.size();i++)
      if(!node[i]->IsLeaf())
        node_lst_vec[node[i]->depth].push_back(node[i]);
    for(int i=0;i<=MAX_DEPTH;i++)
      for(size_t j=0;j<node_lst_vec[i].size();j++)
        node_lst.push_back(node_lst_vec[i][j]);
    n_list[indx]=node_lst;

    indx=3;
    node_lst.clear();
    for(int i=0;i<=MAX_DEPTH;i++)
      node_lst_vec[i].clear();
    for(size_t i=0;i<node.size();i++)
      if(!node[i]->IsLeaf() && !node[i]->IsGhost())
        node_lst_vec[node[i]->depth].push_back(node[i]);
    for(int i=0;i<=MAX_DEPTH;i++)
      for(size_t j=0;j<node_lst_vec[i].size();j++)
        node_lst.push_back(node_lst_vec[i][j]);
    n_list[indx]=node_lst;

    indx=4;
    int src_dof=kernel->ker_dim[0];
    int surf_dof=3+src_dof;
    node_lst.clear();
    for(size_t i=0;i<node.size();i++) {
      if(node[i]->IsLeaf()){
        node_lst.push_back(node[i]);
      }else{
        node[i]->src_value.Resize(0);
        node[i]->surf_value.Resize(0);
      }
    }
    n_list[indx]=node_lst;
    std::vector<Vector<real_t>*>& vec_lst4=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){
      FMM_Node* node=node_lst[i];
      Vector<real_t>& data_vec=node->src_value;
      size_t vec_sz=(node->src_coord.Dim()/3)*src_dof;
      if(data_vec.Dim()!=vec_sz) data_vec.Resize(vec_sz);
      vec_lst4.push_back(&data_vec);
      Vector<real_t>& data_vec2=node->surf_value;
      vec_sz=(node->surf_coord.Dim()/3)*surf_dof;
      if(data_vec2.Dim()!=vec_sz) data_vec2.Resize(vec_sz);
      vec_lst4.push_back(&data_vec2);
    }

    indx=5;
    int trg_dof=kernel->ker_dim[1];
    node_lst.clear();
    for(size_t i=0;i<node.size();i++) {
      if(node[i]->IsLeaf() && !node[i]->IsGhost()){
        node_lst.push_back(node[i]);
      }else{
        node[i]->trg_value.Resize(0);
      }
    }
    n_list[indx]=node_lst;
    std::vector<Vector<real_t>*>& vec_lst5=vec_list[indx];
    for(size_t i=0;i<node_lst.size();i++){
      FMM_Node* node=node_lst[i];
      Vector<real_t>& data_vec=node->trg_value;
      size_t vec_sz=(node->trg_coord.Dim()/3)*trg_dof;
      data_vec.Resize(vec_sz);
      vec_lst5.push_back(&data_vec);
    }
    {
      indx=6;
      node_lst.clear();
      for(size_t i=0;i<node.size();i++){
        if(node[i]->IsLeaf()){
          node_lst.push_back(node[i]);
        }else{
          node[i]->src_coord.Resize(0);
          node[i]->surf_coord.Resize(0);
          node[i]->trg_coord.Resize(0);
        }
      }
      n_list[indx]=node_lst;
      std::vector<Vector<real_t>*>& vec_lst6=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMM_Node* node=node_lst[i];
        {
          Vector<real_t>& data_vec=node->src_coord;
          vec_lst6.push_back(&data_vec);
        }
        {
          Vector<real_t>& data_vec=node->surf_coord;
          vec_lst6.push_back(&data_vec);
        }
        {
          Vector<real_t>& data_vec=node->trg_coord;
          vec_lst6.push_back(&data_vec);
        }
      }
      {
        if(upwd_check_surf.size()==0){
          size_t m=multipole_order;
          upwd_check_surf.resize(MAX_DEPTH);
          upwd_equiv_surf.resize(MAX_DEPTH);
          dnwd_check_surf.resize(MAX_DEPTH);
          dnwd_equiv_surf.resize(MAX_DEPTH);
          for(size_t depth=0;depth<MAX_DEPTH;depth++){
            real_t c[3]={0.0,0.0,0.0};
            upwd_check_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
            upwd_equiv_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
            dnwd_check_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
            dnwd_equiv_surf[depth].Resize((6*(m-1)*(m-1)+2)*3);
            upwd_check_surf[depth]=u_check_surf(m,c,depth);
            upwd_equiv_surf[depth]=u_equiv_surf(m,c,depth);
            dnwd_check_surf[depth]=d_check_surf(m,c,depth);
            dnwd_equiv_surf[depth]=d_equiv_surf(m,c,depth);
          }
        }
        for(size_t depth=0;depth<MAX_DEPTH;depth++){
          vec_lst6.push_back(&upwd_check_surf[depth]);
          vec_lst6.push_back(&upwd_equiv_surf[depth]);
          vec_lst6.push_back(&dnwd_check_surf[depth]);
          vec_lst6.push_back(&dnwd_equiv_surf[depth]);
        }
      }
    }
    if(buff_list.size()<=vec_list.size()) buff_list.resize(vec_list.size()+1);
    for(size_t indx=0;indx<vec_list.size();indx++){
      Matrix<real_t>& buff=buff_list[indx];
      std::vector<Vector<real_t>*>& vec_lst= vec_list[indx];
      bool keep_data=(indx==4 || indx==6);
      size_t n_vec=vec_lst.size();
      if(!n_vec) continue;
      std::vector<size_t> vec_size(n_vec);
      std::vector<size_t> vec_disp(n_vec);
#pragma omp parallel for
      for(size_t i=0;i<n_vec;i++) {
	vec_size[i]=vec_lst[i]->Dim();
      }
      vec_disp[0]=0;
      scan(&vec_size[0],&vec_disp[0],n_vec);
      size_t buff_size=vec_size[n_vec-1]+vec_disp[n_vec-1];
      if(!buff_size) continue;
      if(buff.Dim(0)*buff.Dim(1)<buff_size){
        buff.ReInit(1,buff_size*1.05);
      }
#pragma omp parallel for
      for(size_t i=0;i<n_vec;i++){
        if(vec_size[i]>0){
          memcpy(&buff[0][0]+vec_disp[i],&vec_lst[i][0][0],vec_size[i]*sizeof(real_t));
        }
        vec_lst[i]->Resize(vec_size[i]);
        vec_lst[i]->ReInit3(vec_size[i],&buff[0][0]+vec_disp[i],false);
      }
    }
  }

  void SetupPrecomp(SetupData& setup_data){
    if(setup_data.precomp_data==NULL || setup_data.level>MAX_DEPTH) return;
    Profile::Tic("SetupPrecomp",true,25);
    {
      size_t precomp_offset=0;
      int level=setup_data.level;
      std::vector<char>& precomp_data=*setup_data.precomp_data;
      std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;
      for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){
        Mat_Type& interac_type=interac_type_lst[type_indx];
        PrecompAll(interac_type, level);
        precomp_offset=mat->CompactData(level, interac_type, precomp_data, precomp_offset);
      }
    }
    Profile::Toc();
  }

  void SetupInterac(SetupData& setup_data){
    int level=setup_data.level;
    std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    Matrix<real_t>&  input_data=*setup_data. input_data;
    Matrix<real_t>& output_data=*setup_data.output_data;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector;
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    if(setup_data.precomp_data->size()==0) SetupPrecomp(setup_data);
    Profile::Tic("Interac-Data",true,25);
    {
      std::vector<size_t> interac_mat;
      std::vector<size_t> interac_cnt;
      std::vector<size_t> interac_blk;
      std::vector<size_t>  input_perm;
      std::vector<size_t> output_perm;
      size_t M_dim0=0, M_dim1=0;
      size_t precomp_offset=0;
      size_t buff_size=1024l*1024l*1024l;
      if(n_out && n_in) for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){
        Mat_Type& interac_type=interac_type_lst[type_indx];
        size_t mat_cnt=interacList.ListCount(interac_type);
        Matrix<size_t> precomp_data_offset;
        {
          struct HeaderData{
            size_t total_size;
            size_t      level;
            size_t   mat_cnt ;
            size_t  max_depth;
          };
          std::vector<char>& precomp_data=*setup_data.precomp_data;
          char* indx_ptr=&precomp_data[0]+precomp_offset;
          HeaderData& header=*(HeaderData*)indx_ptr;indx_ptr+=sizeof(HeaderData);
          precomp_data_offset.ReInit(header.mat_cnt,(1+(2+2)*header.max_depth), (size_t*)indx_ptr, false);
          precomp_offset+=header.total_size;
        }
        FMM_Node*** src_interac_list = new FMM_Node** [n_in];
        for (int i=0; i<n_in; i++) {
          src_interac_list[i] = new FMM_Node* [mat_cnt];
          for (int j=0; j<mat_cnt; j++) {
            src_interac_list[i][j] = NULL;
          }
        }
        FMM_Node*** trg_interac_list = new FMM_Node** [n_out];
        for (int i=0; i<n_out; i++) {
          trg_interac_list[i] = new FMM_Node* [mat_cnt];
          for (int j=0; j<mat_cnt; j++) {
            trg_interac_list[i][j] = NULL;
          }
        }
        {
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            if(!nodes_out[i]->IsGhost() && (level==-1 || nodes_out[i]->depth==level)){
              std::vector<FMM_Node*>& lst=nodes_out[i]->interac_list[interac_type];
              for (int l=0; l<lst.size(); l++) {
                trg_interac_list[i][l] = lst[l];
              }
              assert(lst.size()==mat_cnt);
            }
          }
        }
        {
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            for(size_t j=0;j<mat_cnt;j++)
            if(trg_interac_list[i][j]!=NULL){
              trg_interac_list[i][j]->node_id=n_in;
            }
          }
#pragma omp parallel for
          for(size_t i=0;i<n_in ;i++) nodes_in[i]->node_id=i;
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            for(size_t j=0;j<mat_cnt;j++){
              if(trg_interac_list[i][j]!=NULL){
                if(trg_interac_list[i][j]->node_id==n_in){
                  trg_interac_list[i][j]=NULL;
                }else{
                  src_interac_list[trg_interac_list[i][j]->node_id][j]=nodes_out[i];
                }
              }
            }
          }
        }
        Matrix<size_t> interac_dsp(n_out,mat_cnt);
        std::vector<size_t> interac_blk_dsp(1,0);
        {
	  Matrix<real_t>& M0 = interacList.ClassMat(level, interac_type_lst[0], 0);
          M_dim0=M0.Dim(0); M_dim1=M0.Dim(1);
        }
        {
          size_t vec_size=(M_dim0+M_dim1)*sizeof(real_t);
          for(size_t j=0;j<mat_cnt;j++){
            size_t vec_cnt=0;
            for(size_t i=0;i<n_out;i++){
              if(trg_interac_list[i][j]!=NULL) vec_cnt++;
            }
            if(buff_size<vec_cnt*vec_size)
              buff_size=vec_cnt*vec_size;
          }
          size_t interac_dsp_=0;
          for(size_t j=0;j<mat_cnt;j++){
            for(size_t i=0;i<n_out;i++){
              interac_dsp[i][j]=interac_dsp_;
              if(trg_interac_list[i][j]!=NULL) interac_dsp_++;
            }
            if(interac_dsp_*vec_size>buff_size) {
              interac_blk.push_back(j-interac_blk_dsp.back());
              interac_blk_dsp.push_back(j);

              size_t offset=interac_dsp[0][j];
              for(size_t i=0;i<n_out;i++) interac_dsp[i][j]-=offset;
              interac_dsp_-=offset;
              assert(interac_dsp_*vec_size<=buff_size);
            }
            interac_mat.push_back(precomp_data_offset[interacList.InteracClass(interac_type,j)][0]);
            interac_cnt.push_back(interac_dsp_-interac_dsp[0][j]);
          }
          interac_blk.push_back(mat_cnt-interac_blk_dsp.back());
          interac_blk_dsp.push_back(mat_cnt);
        }
        {
          size_t vec_size=M_dim0;
          for(size_t i=0;i<n_out;i++) nodes_out[i]->node_id=i;
          for(size_t k=1;k<interac_blk_dsp.size();k++){
            for(size_t i=0;i<n_in ;i++){
              for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
                FMM_Node* trg_node=src_interac_list[i][j];
                if(trg_node!=NULL && trg_node->node_id<n_out){
                  size_t depth=trg_node->depth;
                  input_perm .push_back(precomp_data_offset[j][1+4*depth+0]);
                  input_perm .push_back(precomp_data_offset[j][1+4*depth+1]);
                  input_perm .push_back(interac_dsp[trg_node->node_id][j]*vec_size*sizeof(real_t));
                  input_perm .push_back((size_t)(& input_vector[i][0][0]- input_data[0]));
                  assert(input_vector[i]->Dim()==vec_size);
                }
              }
            }
          }
        }
        {
          size_t vec_size=M_dim1;
          for(size_t k=1;k<interac_blk_dsp.size();k++){
            for(size_t i=0;i<n_out;i++){
              for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
                if(trg_interac_list[i][j]!=NULL){
                  size_t depth=nodes_out[i]->depth;
                  output_perm.push_back(precomp_data_offset[j][1+4*depth+2]);
                  output_perm.push_back(precomp_data_offset[j][1+4*depth+3]);
                  output_perm.push_back(interac_dsp[               i ][j]*vec_size*sizeof(real_t));
                  output_perm.push_back((size_t)(&output_vector[i][0][0]-output_data[0]));
                }
              }
            }
          }
        }
        for (int i=0; i<n_in; i++) {
          delete[] src_interac_list[i];
        }
        delete[] src_interac_list;
        for (int i=0; i<n_out; i++) {
          delete[] trg_interac_list[i];
        }
        delete[] trg_interac_list;
      }

      if(dev_buffer.Dim()<buff_size) dev_buffer.Resize(buff_size);
      setup_data.M_dim0 = M_dim0;
      setup_data.M_dim1 = M_dim1;
      setup_data.interac_blk = interac_blk;
      setup_data.interac_cnt = interac_cnt;
      setup_data.interac_mat = interac_mat;
      setup_data.input_perm = input_perm;
      setup_data.output_perm = output_perm;
    }
    Profile::Toc();
  }


  void PtSetup(SetupData& setup_data, ptSetupData* data_){
    ptSetupData& data=*(ptSetupData*)data_;
    if(data.pt_interac_data.interac_cnt.Dim()){
      InteracData& intdata=data.pt_interac_data;
      Vector<size_t>  cnt;
      Vector<size_t>& dsp=intdata.interac_cst;
      cnt.Resize(intdata.interac_cnt.Dim());
      dsp.Resize(intdata.interac_dsp.Dim());
#pragma omp parallel for
      for(size_t trg=0;trg<cnt.Dim();trg++){
        size_t trg_cnt=data.trg_coord.cnt[trg];
        cnt[trg]=0;
        for(size_t i=0;i<intdata.interac_cnt[trg];i++){
          size_t int_id=intdata.interac_dsp[trg]+i;
          size_t src=intdata.in_node[int_id];
          size_t src_cnt=data.src_coord.cnt[src];
          size_t srf_cnt=data.srf_coord.cnt[src];
          cnt[trg]+=(src_cnt+srf_cnt)*trg_cnt;
        }
      }
      dsp[0]=cnt[0];
      scan(&cnt[0],&dsp[0],dsp.Dim());
    }
    setup_data.pt_setup_data = data;
    {
      size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(real_t);
      if(dev_buffer.Dim()<n) dev_buffer.Resize(n);
    }
  }

  void Source2UpSetup(SetupData& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level) {
    if(!multipole_order) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_s2m;
      setup_data. input_data=&buff[4];
      setup_data.output_data=&buff[0];
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[4];
      std::vector<FMM_Node*>& nodes_out=n_list[0];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++)
        if((nodes_in [i]->depth==level || level==-1)
  	 && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim())
  	 && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++)
        if((nodes_out[i]->depth==level || level==-1)
  	 && (nodes_out[i]->src_coord.Dim() || nodes_out[i]->surf_coord.Dim())
  	 && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        nodes[i]->node_id=i;
        Vector<real_t>& coord_vec=nodes[i]->src_coord;
        Vector<real_t>& value_vec=nodes[i]->src_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->surf_coord;
        Vector<real_t>& value_vec=nodes[i]->surf_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=upwd_check_surf[nodes[i]->depth];
        Vector<real_t>& value_vec=(nodes[i]->FMMData())->upward_equiv;

//std::cout << value_vec.Dim() << std::endl;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      const Kernel* ker=kernel->k_m2m;
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+2];
        std::vector<real_t>& scal_exp=ker->trg_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+3];
        std::vector<real_t>& scal_exp=ker->src_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=S2U_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+0.5*s)+(0+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+0.5*s)+(0+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+0.5*s)+(0+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
        pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
        if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
          data.pt_interac_data.M[2]=mat->Mat(level, UC2UE0_Type, 0);
          data.pt_interac_data.M[3]=mat->Mat(level, UC2UE1_Type, 0);
        }else{
          data.pt_interac_data.M[2].ReInit(0,0);
          data.pt_interac_data.M[3].ReInit(0,0);
        }
      }
    }
    PtSetup(setup_data, &data);
  }


  void Up2UpSetup(SetupData& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    {
      setup_data.level=level;
      setup_data.kernel=kernel->k_m2m;
      setup_data.interac_type.resize(1);
      setup_data.interac_type[0]=U2U_Type;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[0];
      std::vector<FMM_Node*>& nodes_in =n_list[0];
      std::vector<FMM_Node*>& nodes_out=n_list[0];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((nodes_in [i]->depth==level+1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((nodes_out[i]->depth==level  ) && nodes_out[i]->pt_cnt[0]) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(nodes_in [i]->FMMData())->upward_equiv);
    for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&(nodes_out[i]->FMMData())->upward_equiv);
    SetupInterac(setup_data);
  }


  void BuildInteracLists() {
    std::vector<FMM_Node*> n_list_src;
    std::vector<FMM_Node*> n_list_trg;
    {
      std::vector<FMM_Node*>& nodes=GetNodeList();
      for(size_t i=0;i<nodes.size();i++){
        if(!nodes[i]->IsGhost() && nodes[i]->pt_cnt[0]){
          n_list_src.push_back(nodes[i]);
        }
        if(!nodes[i]->IsGhost() && nodes[i]->pt_cnt[1]){
          n_list_trg.push_back(nodes[i]);
        }
      }
    }
    size_t node_cnt=std::max(n_list_src.size(),n_list_trg.size());
    std::vector<Mat_Type> type_lst;
    std::vector<std::vector<FMM_Node*>*> type_node_lst;
    type_lst.push_back(S2U_Type); type_node_lst.push_back(&n_list_src);
    type_lst.push_back(U2U_Type); type_node_lst.push_back(&n_list_src);
    type_lst.push_back(D2D_Type); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(D2T_Type); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(U0_Type ); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(U1_Type ); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(U2_Type ); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(W_Type  ); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(X_Type  ); type_node_lst.push_back(&n_list_trg);
    type_lst.push_back(V1_Type ); type_node_lst.push_back(&n_list_trg);
    std::vector<size_t> interac_cnt(type_lst.size());
    std::vector<size_t> interac_dsp(type_lst.size(),0);
    for(size_t i=0;i<type_lst.size();i++){
      interac_cnt[i]=interacList.ListCount(type_lst[i]);
    }
    scan(&interac_cnt[0],&interac_dsp[0],type_lst.size());
    int omp_p=omp_get_max_threads();
#pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      for(size_t k=0;k<type_lst.size();k++){
        std::vector<FMM_Node*>& n_list=*type_node_lst[k];
        size_t a=(n_list.size()*(j  ))/omp_p;
        size_t b=(n_list.size()*(j+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* n=n_list[i];
          n->interac_list[type_lst[k]].resize(interac_cnt[k]);
          interacList.BuildList(n,type_lst[k]);
        }
      }
    }
  }



  void V_ListSetup(SetupData&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    if(level==0) return;
    {
      setup_data.level=level;
      setup_data.kernel=kernel->k_m2l;
      setup_data.interac_type.resize(1);
      setup_data.interac_type[0]=V1_Type;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[1];
      std::vector<FMM_Node*>& nodes_in =n_list[2];
      std::vector<FMM_Node*>& nodes_out=n_list[3];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((nodes_in [i]->depth==level-1 || level==-1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((nodes_out[i]->depth==level-1 || level==-1) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(nodes_in[i]->Child(0)->FMMData())->upward_equiv);
    for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&(nodes_out[i]->Child(0)->FMMData())->dnward_equiv);
    real_t eps=1e-10;
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    Profile::Tic("Interac-Data",true,25);
    if(n_out>0 && n_in >0){
      size_t precomp_offset=0;
      Mat_Type& interac_type=setup_data.interac_type[0];
      size_t mat_cnt=interacList.ListCount(interac_type);
      Matrix<size_t> precomp_data_offset;
      std::vector<real_t*> precomp_mat;
      {
        for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
          Matrix<real_t>& M = mat->Mat(level, interac_type, mat_id);
          precomp_mat.push_back(&M[0][0]);
        }
      }
      size_t m=multipole_order;
      size_t ker_dim0=setup_data.kernel->ker_dim[0];
      size_t ker_dim1=setup_data.kernel->ker_dim[1];
      size_t fftsize;
      {
        size_t n1=m*2;
        size_t n2=n1*n1;
        size_t n3_=n2*(n1/2+1);
        size_t chld_cnt=1UL<<3;
        fftsize=2*n3_*chld_cnt;
      }
      int omp_p=omp_get_max_threads();
      size_t buff_size=1024l*1024l*1024l;
      size_t n_blk0=2*fftsize*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(real_t)/buff_size;
      if(n_blk0==0) n_blk0=1;
      std::vector<std::vector<size_t> >  fft_vec(n_blk0);
      std::vector<std::vector<size_t> > ifft_vec(n_blk0);
      std::vector<std::vector<real_t> >  fft_scl(n_blk0);
      std::vector<std::vector<real_t> > ifft_scl(n_blk0);
      std::vector<std::vector<size_t> > interac_vec(n_blk0);
      std::vector<std::vector<size_t> > interac_dsp(n_blk0);
      {
        Matrix<real_t>&  input_data=*setup_data. input_data;
        Matrix<real_t>& output_data=*setup_data.output_data;
        std::vector<std::vector<FMM_Node*> > nodes_blk_in (n_blk0);
        std::vector<std::vector<FMM_Node*> > nodes_blk_out(n_blk0);
        std::vector<real_t> src_scal=kernel->k_m2l->src_scal;
        std::vector<real_t> trg_scal=kernel->k_m2l->trg_scal;

        for(size_t i=0;i<n_in;i++) nodes_in[i]->node_id=i;
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          size_t blk0_start=(n_out* blk0   )/n_blk0;
          size_t blk0_end  =(n_out*(blk0+1))/n_blk0;
          std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
          {
            std::set<FMM_Node*> nodes_in;
            for(size_t i=blk0_start;i<blk0_end;i++){
              nodes_out_.push_back(nodes_out[i]);
              std::vector<FMM_Node*>& lst=nodes_out[i]->interac_list[interac_type];
              for(size_t k=0;k<mat_cnt;k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
            }
            for(typename std::set<FMM_Node*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++){
              nodes_in_.push_back(*node);
            }
            size_t  input_dim=nodes_in_ .size()*ker_dim0*fftsize;
            size_t output_dim=nodes_out_.size()*ker_dim1*fftsize;
            size_t buffer_dim=2*(ker_dim0+ker_dim1)*fftsize*omp_p;
            if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(real_t))
              buff_size=(input_dim + output_dim + buffer_dim)*sizeof(real_t);
          }
          {
            for(size_t i=0;i<nodes_in_ .size();i++) fft_vec[blk0].push_back((size_t)(& input_vector[nodes_in_[i]->node_id][0][0]- input_data[0]));
            for(size_t i=0;i<nodes_out_.size();i++)ifft_vec[blk0].push_back((size_t)(&output_vector[blk0_start   +     i ][0][0]-output_data[0]));
            size_t scal_dim0=src_scal.size();
            size_t scal_dim1=trg_scal.size();
            fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
            ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
            for(size_t i=0;i<nodes_in_ .size();i++){
              size_t depth=nodes_in_[i]->depth+1;
              for(size_t j=0;j<scal_dim0;j++){
                fft_scl[blk0][i*scal_dim0+j]=powf(2.0, src_scal[j]*depth);
              }
            }
            for(size_t i=0;i<nodes_out_.size();i++){
              size_t depth=nodes_out_[i]->depth+1;
              for(size_t j=0;j<scal_dim1;j++){
                ifft_scl[blk0][i*scal_dim1+j]=powf(2.0, trg_scal[j]*depth);
              }
            }
          }
        }
        for(size_t blk0=0;blk0<n_blk0;blk0++){
          std::vector<FMM_Node*>& nodes_in_ =nodes_blk_in [blk0];
          std::vector<FMM_Node*>& nodes_out_=nodes_blk_out[blk0];
          for(size_t i=0;i<nodes_in_.size();i++) nodes_in_[i]->node_id=i;
          {
            size_t n_blk1=nodes_out_.size()*sizeof(real_t)/CACHE_SIZE;
            if(n_blk1==0) n_blk1=1;
            size_t interac_dsp_=0;
            for(size_t blk1=0;blk1<n_blk1;blk1++){
              size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
              size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
              for(size_t k=0;k<mat_cnt;k++){
                for(size_t i=blk1_start;i<blk1_end;i++){
                  std::vector<FMM_Node*>& lst=nodes_out_[i]->interac_list[interac_type];
                  if(lst[k]!=NULL && lst[k]->pt_cnt[0]){
                    interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0);
                    interac_vec[blk0].push_back(    i          *fftsize*ker_dim1);
                    interac_dsp_++;
                  }
                }
                interac_dsp[blk0].push_back(interac_dsp_);
              }
            }
          }
        }
      }
      setup_data.vlist_data.buff_size   = buff_size;
      setup_data.vlist_data.m           = m;
      setup_data.vlist_data.n_blk0      = n_blk0;
      setup_data.vlist_data.precomp_mat = precomp_mat;
      setup_data.vlist_data.fft_vec     = fft_vec;
      setup_data.vlist_data.ifft_vec    = ifft_vec;
      setup_data.vlist_data.fft_scl     = fft_scl;
      setup_data.vlist_data.ifft_scl    = ifft_scl;
      setup_data.vlist_data.interac_vec = interac_vec;
      setup_data.vlist_data.interac_dsp = interac_dsp;
    }
    Profile::Toc();
  }

  void FFT_UpEquiv(size_t m, std::vector<size_t>& fft_vec, std::vector<real_t>& fft_scal,
		   Vector<real_t>& input_data, Vector<real_t>& output_data, Vector<real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        real_t c[3]={0,0,0};
        Vector<real_t> surf=surface(m, c, (real_t)(m-1), 0);
        map.Resize(surf.Dim()/3);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(m-1-surf[i*3+2]+0.5))*n2;
      }
    }
    {
      if(!vlist_fft_flag){
        int err, nnn[3]={(int)n1,(int)n1,(int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        vlist_fftplan = fft_plan_many_dft_r2c(3,nnn,chld_cnt,
					      (real_t*)fftw_in, NULL, 1, n3,
					      (fft_complex*)(fftw_out),NULL, 1, n3_,
					      FFTW_ESTIMATE);
        free(fftw_in );
        free(fftw_out);
        vlist_fft_flag=true;
      }
    }
    {
      size_t n_in = fft_vec.size();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_in*(pid  ))/omp_p;
        size_t node_end  =(n_in*(pid+1))/omp_p;
        Vector<real_t> buffer(fftsize_in, &buffer_[fftsize_in*pid], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Matrix<real_t>  upward_equiv(chld_cnt,n,&input_data[0] + fft_vec[node_idx],false);
          Vector<real_t> upward_equiv_fft(fftsize_in, &output_data[fftsize_in *node_idx], false);
          upward_equiv_fft.SetZero();
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            int j1=0;
            for(int j0=0;j0<(int)chld_cnt;j0++)
              upward_equiv_fft[idx+j0*n3]=upward_equiv[j0][k]*fft_scal[node_idx];
          }
          fft_execute_dft_r2c(vlist_fftplan, (real_t*)&upward_equiv_fft[0], (fft_complex*)&buffer[0]);
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            upward_equiv_fft[2*(chld_cnt*j+k)+0]=buffer[2*(n3_*k+j)+0];
            upward_equiv_fft[2*(chld_cnt*j+k)+1]=buffer[2*(n3_*k+j)+1];
          }
        }
      }
    }
  }

  void FFT_Check2Equiv(size_t m, std::vector<size_t>& ifft_vec, std::vector<real_t>& ifft_scal,
		       Vector<real_t>& input_data, Vector<real_t>& output_data, Vector<real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_out=2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        real_t c[3]={0,0,0};
        Vector<real_t> surf=surface(m, c, (real_t)(m-1), 0);
        map.Resize(surf.Dim()/3);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(m*2-0.5-surf[i*3+2]))*n2;
      }
    }
    {
      if(!vlist_ifft_flag){
        int err, nnn[3]={(int)n1,(int)n1,(int)n1};
        real_t *fftw_in, *fftw_out;
        err = posix_memalign((void**)&fftw_in,  MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
        err = posix_memalign((void**)&fftw_out, MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
        vlist_ifftplan = fft_plan_many_dft_c2r(3,nnn,chld_cnt,
					       (fft_complex*)fftw_in, NULL, 1, n3_,
					       (real_t*)(fftw_out),NULL, 1, n3,
					       FFTW_ESTIMATE);
        free(fftw_in);
        free(fftw_out);
        vlist_ifft_flag=true;
      }
    }
    {
      assert(buffer_.Dim()>=2*fftsize_out*omp_p);
      size_t n_out=ifft_vec.size();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_out*(pid  ))/omp_p;
        size_t node_end  =(n_out*(pid+1))/omp_p;
        Vector<real_t> buffer0(fftsize_out, &buffer_[fftsize_out*(2*pid+0)], false);
        Vector<real_t> buffer1(fftsize_out, &buffer_[fftsize_out*(2*pid+1)], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Vector<real_t> dnward_check_fft(fftsize_out, &input_data[fftsize_out*node_idx], false);
          Vector<real_t> dnward_equiv(n*chld_cnt,&output_data[0] + ifft_vec[node_idx],false);
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            buffer0[2*(n3_*k+j)+0]=dnward_check_fft[2*(chld_cnt*j+k)+0];
            buffer0[2*(n3_*k+j)+1]=dnward_check_fft[2*(chld_cnt*j+k)+1];
          }
          fft_execute_dft_c2r(vlist_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            for(int j0=0;j0<(int)chld_cnt;j0++)
              dnward_equiv[n*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
          }
        }
      }
    }
  }

  void Down2DownSetup(SetupData& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    {
      setup_data.level=level;
      setup_data.kernel=kernel->k_l2l;
      setup_data.interac_type.resize(1);
      setup_data.interac_type[0]=D2D_Type;
      setup_data. input_data=&buff[1];
      setup_data.output_data=&buff[1];
      std::vector<FMM_Node*>& nodes_in =n_list[1];
      std::vector<FMM_Node*>& nodes_out=n_list[1];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((nodes_in [i]->depth==level-1) && nodes_in [i]->pt_cnt[1]) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((nodes_out[i]->depth==level  ) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    std::vector<Vector<real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
    std::vector<Vector<real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
    for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&(nodes_in[i]->FMMData())->dnward_equiv);
    for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&(nodes_out[i]->FMMData())->dnward_equiv);
    SetupInterac(setup_data);
  }

  void X_ListSetup(SetupData&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_s2l;
      setup_data. input_data=&buff[4];
      setup_data.output_data=&buff[1];
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[4];
      std::vector<FMM_Node*>& nodes_out=n_list[1];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) &&  nodes_in [i]->IsLeaf ()) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((level==0 || level==-1) &&  nodes_out[i]->pt_cnt[1]                                          && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        ((FMM_Node*)nodes[i])->node_id=i;
        Vector<real_t>& coord_vec=nodes[i]->src_coord;
        Vector<real_t>& value_vec=nodes[i]->src_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->surf_coord;
        Vector<real_t>& value_vec=nodes[i]->surf_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=dnwd_check_surf[nodes[i]->depth];
        Vector<real_t>& value_vec=(nodes[i]->FMMData())->dnward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid];
        std::vector<size_t>& scal_idx   =scal_idx_[tid];
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          if(tnode->IsLeaf() && tnode->pt_cnt[1]<=Nsrf){
            interac_cnt.push_back(0);
            continue;
          }
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=X_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(0+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(0+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(0+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }
  void W_ListSetup(SetupData&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_m2t;
      setup_data. input_data=&buff[0];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[0];
      std::vector<FMM_Node*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((level==0 || level==-1) && nodes_in [i]->pt_cnt[0]                                                            ) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((level==0 || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        ((FMM_Node*)nodes[i])->node_id=i;
        Vector<real_t>& coord_vec=upwd_equiv_surf[nodes[i]->depth];
        Vector<real_t>& value_vec=(nodes[i]->FMMData())->upward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        coord.dsp[i]=0;
        coord.cnt[i]=0;
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=W_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0){
              }else if(snode->IsLeaf() && snode->pt_cnt[0]<=Nsrf) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.25*s-(0+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(0+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(0+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  void U_ListSetup(SetupData& setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_s2t;
      setup_data. input_data=&buff[4];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[4];
      std::vector<FMM_Node*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++)
        if((level==0 || level==-1)
  	 && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim())
  	 && nodes_in [i]->IsLeaf()                            ) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++)
        if((level==0 || level==-1)
  	 && (nodes_out[i]->trg_coord.Dim()                                  )
  	 && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        nodes[i]->node_id=i;
        Vector<real_t>& coord_vec=nodes[i]->src_coord;
        Vector<real_t>& value_vec=nodes[i]->src_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->surf_coord;
        Vector<real_t>& value_vec=nodes[i]->surf_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      size_t m=multipole_order;
      size_t Nsrf=(6*(m-1)*(m-1)+2);
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=U0_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U1_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*1.0*s-(scoord[0]+0.5*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*1.0*s-(scoord[1]+0.5*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*1.0*s-(scoord[2]+0.5*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=U2_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=X_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            if(tnode->pt_cnt[1]<=Nsrf)
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          {
            Mat_Type type=W_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0) continue;
              if(snode->pt_cnt[0]> Nsrf) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
    }
    PtSetup(setup_data, &data);
  }
  void Down2TargetSetup(SetupData&  setup_data, std::vector<Matrix<real_t> >& buff, std::vector<std::vector<FMM_Node*> >& n_list, int level){
    if(!multipole_order) return;
    {
      setup_data. level=level;
      setup_data.kernel=kernel->k_l2t;
      setup_data. input_data=&buff[1];
      setup_data.output_data=&buff[5];
      setup_data. coord_data=&buff[6];
      std::vector<FMM_Node*>& nodes_in =n_list[1];
      std::vector<FMM_Node*>& nodes_out=n_list[5];
      setup_data.nodes_in .clear();
      setup_data.nodes_out.clear();
      for(size_t i=0;i<nodes_in .size();i++) if((nodes_in [i]->depth==level || level==-1) && nodes_in [i]->trg_coord.Dim() && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
      for(size_t i=0;i<nodes_out.size();i++) if((nodes_out[i]->depth==level || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
    }
    ptSetupData data;
    data. level=setup_data. level;
    data.kernel=setup_data.kernel;
    std::vector<FMM_Node*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMM_Node*>& nodes_out=setup_data.nodes_out;
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.src_coord;
      PackedData& value=data.src_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        nodes[i]->node_id=i;
        Vector<real_t>& coord_vec=dnwd_equiv_surf[nodes[i]->depth];
        Vector<real_t>& value_vec=nodes[i]->FMMData()->dnward_equiv;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_in;
      PackedData& coord=data.srf_coord;
      PackedData& value=data.srf_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data. input_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        coord.dsp[i]=0;
        coord.cnt[i]=0;
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
    {
      std::vector<FMM_Node*>& nodes=nodes_out;
      PackedData& coord=data.trg_coord;
      PackedData& value=data.trg_value;
      coord.ptr=setup_data. coord_data;
      value.ptr=setup_data.output_data;
      coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
      value.len=value.ptr->Dim(0)*value.ptr->Dim(1);
      coord.cnt.Resize(nodes.size());
      coord.dsp.Resize(nodes.size());
      value.cnt.Resize(nodes.size());
      value.dsp.Resize(nodes.size());
#pragma omp parallel for
      for(size_t i=0;i<nodes.size();i++){
        Vector<real_t>& coord_vec=nodes[i]->trg_coord;
        Vector<real_t>& value_vec=nodes[i]->trg_value;
        if(coord_vec.Dim()){
          coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
          assert(coord.dsp[i]<coord.len);
          coord.cnt[i]=coord_vec.Dim();
        }else{
          coord.dsp[i]=0;
          coord.cnt[i]=0;
        }
        if(value_vec.Dim()){
          value.dsp[i]=&value_vec[0]-value.ptr[0][0];
          assert(value.dsp[i]<value.len);
          value.cnt[i]=value_vec.Dim();
        }else{
          value.dsp[i]=0;
          value.cnt[i]=0;
        }
      }
    }
    {
      int omp_p=omp_get_max_threads();
      std::vector<std::vector<size_t> > in_node_(omp_p);
      std::vector<std::vector<size_t> > scal_idx_(omp_p);
      std::vector<std::vector<real_t> > coord_shift_(omp_p);
      std::vector<std::vector<size_t> > interac_cnt_(omp_p);
      const Kernel* ker=kernel->k_l2l;
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+0];
        std::vector<real_t>& scal_exp=ker->trg_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){
        Vector<real_t>& scal=data.pt_interac_data.scal[l*4+1];
        std::vector<real_t>& scal_exp=ker->src_scal;
        scal.Resize(scal_exp.size());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=powf(2.0,-scal_exp[i]*l);
        }
      }
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        std::vector<size_t>& in_node    =in_node_[tid]    ;
        std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
        std::vector<real_t>& coord_shift=coord_shift_[tid];
        std::vector<size_t>& interac_cnt=interac_cnt_[tid];
        size_t a=(nodes_out.size()*(tid+0))/omp_p;
        size_t b=(nodes_out.size()*(tid+1))/omp_p;
        for(size_t i=a;i<b;i++){
          FMM_Node* tnode=nodes_out[i];
          real_t s=powf(0.5,tnode->depth);
          size_t interac_cnt_=0;
          {
            Mat_Type type=D2T_Type;
            std::vector<FMM_Node*>& intlst=tnode->interac_list[type];
            for(size_t j=0;j<intlst.size();j++) if(intlst[j]){
              FMM_Node* snode=intlst[j];
              size_t snode_id=snode->node_id;
              if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
              in_node.push_back(snode_id);
              scal_idx.push_back(snode->depth);
              {
                const int* rel_coord=interacList.RelativeCoord(type,j);
                const real_t* scoord=snode->Coord();
                const real_t* tcoord=tnode->Coord();
                real_t shift[3];
                shift[0]=rel_coord[0]*0.5*s-(0+0.5*s)+(tcoord[0]+0.5*s);
                shift[1]=rel_coord[1]*0.5*s-(0+0.5*s)+(tcoord[1]+0.5*s);
                shift[2]=rel_coord[2]*0.5*s-(0+0.5*s)+(tcoord[2]+0.5*s);
                coord_shift.push_back(shift[0]);
                coord_shift.push_back(shift[1]);
                coord_shift.push_back(shift[2]);
              }
              interac_cnt_++;
            }
          }
          interac_cnt.push_back(interac_cnt_);
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
	CopyVec(in_node_,pt_interac_data.in_node);
	CopyVec(scal_idx_,pt_interac_data.scal_idx);
	CopyVec(coord_shift_,pt_interac_data.coord_shift);
	CopyVec(interac_cnt_,pt_interac_data.interac_cnt);
        {
          pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
          pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
          dsp.Resize(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
          scan(&cnt[0],&dsp[0],dsp.Dim());
        }
      }
      {
        InteracData& pt_interac_data=data.pt_interac_data;
        pvfmm::Vector<size_t>& cnt=pt_interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=pt_interac_data.interac_dsp;
        if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
          data.pt_interac_data.M[0]=mat->Mat(level, DC2DE0_Type, 0);
          data.pt_interac_data.M[1]=mat->Mat(level, DC2DE1_Type, 0);
        }else{
          data.pt_interac_data.M[0].ReInit(0,0);
          data.pt_interac_data.M[1].ReInit(0,0);
        }
      }
    }
    PtSetup(setup_data, &data);
  }

  /* below are traversal functions */
private:
  void EvalListPts(SetupData& setup_data) {
    if(setup_data.kernel->ker_dim[0]*setup_data.kernel->ker_dim[1]==0) return;
    bool have_gpu=false;
    Profile::Tic("Host2Device",false,25);
    char* dev_buff = dev_buffer.data_ptr;
    Profile::Toc();
    Profile::Tic("DeviceComp",false,20);
    int lock_idx=-1;
    int wait_lock_idx=-1;
    {
      ptSetupData data = setup_data.pt_setup_data;
      {
        InteracData& intdata = data.pt_interac_data;
        int omp_p=omp_get_max_threads();
#pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          Matrix<real_t> src_coord, src_value;
          Matrix<real_t> srf_coord, srf_value;
          Matrix<real_t> trg_coord, trg_value;
          Vector<real_t> buff;
          {
            size_t n=setup_data.output_data->Dim(0)*setup_data.output_data->Dim(1)*sizeof(real_t);
            size_t thread_buff_size=n/sizeof(real_t)/omp_p;
            buff.ReInit3(thread_buff_size, (real_t*)(dev_buff+tid*thread_buff_size*sizeof(real_t)), false);
          }
          size_t vcnt=0;
          std::vector<Matrix<real_t> > vbuff(6);
          {
            size_t vdim_=0, vdim[6];
            for(size_t indx=0;indx<6;indx++){
              vdim[indx]=0;
              switch(indx){
                case 0:
                  vdim[indx]=intdata.M[0].Dim(0); break;
                case 1:
                  assert(intdata.M[0].Dim(1)==intdata.M[1].Dim(0));
                  vdim[indx]=intdata.M[0].Dim(1); break;
                case 2:
                  vdim[indx]=intdata.M[1].Dim(1); break;
                case 3:
                  vdim[indx]=intdata.M[2].Dim(0); break;
                case 4:
                  assert(intdata.M[2].Dim(1)==intdata.M[3].Dim(0));
                  vdim[indx]=intdata.M[2].Dim(1); break;
                case 5:
                  vdim[indx]=intdata.M[3].Dim(1); break;
                default:
                  vdim[indx]=0; break;
              }
              vdim_+=vdim[indx];
            }
            if(vdim_){
              vcnt=buff.Dim()/vdim_/2;
              assert(vcnt>0);
            }
            for(size_t indx=0;indx<6;indx++){
              vbuff[indx].ReInit(vcnt,vdim[indx],&buff[0],false);
              buff.ReInit3(buff.Dim()-vdim[indx]*vcnt, &buff[vdim[indx]*vcnt], false);
            }
          }
          size_t trg_a=0, trg_b=0;
          if(intdata.interac_cst.Dim()){
            Vector<size_t>& interac_cst=intdata.interac_cst;
            size_t cost=interac_cst[interac_cst.Dim()-1];
            trg_a=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+0))/omp_p)-&interac_cst[0]+1;
            trg_b=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+1))/omp_p)-&interac_cst[0]+1;
            if(tid==omp_p-1) trg_b=interac_cst.Dim();
            if(tid==0) trg_a=0;
          }
          for(size_t trg0=trg_a;trg0<trg_b;){
            size_t trg1_max=1;
            if(vcnt){
              size_t interac_cnt=intdata.interac_cnt[trg0];
              while(trg0+trg1_max<trg_b){
                interac_cnt+=intdata.interac_cnt[trg0+trg1_max];
                if(interac_cnt>vcnt){
                  interac_cnt-=intdata.interac_cnt[trg0+trg1_max];
                  break;
                }
                trg1_max++;
              }
              assert(interac_cnt<=vcnt);
              for(size_t k=0;k<6;k++){
                if(vbuff[k].Dim(0)*vbuff[k].Dim(1)){
                  vbuff[k].ReInit(interac_cnt,vbuff[k].Dim(1),vbuff[k][0],false);
                }
              }
            }else{
              trg1_max=trg_b-trg0;
            }
            if(intdata.M[0].Dim(0) && intdata.M[0].Dim(1) && intdata.M[1].Dim(0) && intdata.M[1].Dim(1)){
              size_t interac_idx=0;
              for(size_t trg1=0;trg1<trg1_max;trg1++){
                size_t trg=trg0+trg1;
                for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                  size_t int_id=intdata.interac_dsp[trg]+i;
                  size_t src=intdata.in_node[int_id];
                  src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                  {
                    size_t vdim=vbuff[0].Dim(1);
                    assert(src_value.Dim(1)==vdim);
                    for(size_t j=0;j<vdim;j++) vbuff[0][interac_idx][j]=src_value[0][j];
                  }
                  size_t scal_idx=intdata.scal_idx[int_id];
                  {
                    Matrix<real_t>& vec=vbuff[0];
                    Vector<real_t>& scal=intdata.scal[scal_idx*4+0];
                    size_t scal_dim=scal.Dim();
                    if(scal_dim){
                      size_t vdim=vec.Dim(1);
                      for(size_t j=0;j<vdim;j+=scal_dim){
                        for(size_t k=0;k<scal_dim;k++){
                          vec[interac_idx][j+k]*=scal[k];
                        }
                      }
                    }
                  }
                  interac_idx++;
                }
              }
              Matrix<real_t>::GEMM(vbuff[1],vbuff[0],intdata.M[0]);
              Matrix<real_t>::GEMM(vbuff[2],vbuff[1],intdata.M[1]);
              interac_idx=0;
              for(size_t trg1=0;trg1<trg1_max;trg1++){
                size_t trg=trg0+trg1;
                for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                  size_t int_id=intdata.interac_dsp[trg]+i;
                  size_t scal_idx=intdata.scal_idx[int_id];
                  {
                    Matrix<real_t>& vec=vbuff[2];
                    Vector<real_t>& scal=intdata.scal[scal_idx*4+1];
                    size_t scal_dim=scal.Dim();
                    if(scal_dim){
                      size_t vdim=vec.Dim(1);
                      for(size_t j=0;j<vdim;j+=scal_dim){
                        for(size_t k=0;k<scal_dim;k++){
                          vec[interac_idx][j+k]*=scal[k];
                        }
                      }
                    }
                  }
                  interac_idx++;
                }
              }
            }
            if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){
              size_t vdim=vbuff[3].Dim(0)*vbuff[3].Dim(1);
              for(size_t i=0;i<vdim;i++) vbuff[3][0][i]=0;
            }
            {
              size_t interac_idx=0;
              for(size_t trg1=0;trg1<trg1_max;trg1++){
                size_t trg=trg0+trg1;
                trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
                trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
                for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                  size_t int_id=intdata.interac_dsp[trg]+i;
                  size_t src=intdata.in_node[int_id];
                  src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
                  src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                  srf_coord.ReInit(1, data.srf_coord.cnt[src], &data.srf_coord.ptr[0][0][data.srf_coord.dsp[src]], false);
                  srf_value.ReInit(1, data.srf_value.cnt[src], &data.srf_value.ptr[0][0][data.srf_value.dsp[src]], false);
                  real_t* vbuff2_ptr=(vbuff[2].Dim(0)*vbuff[2].Dim(1)?vbuff[2][interac_idx]:src_value[0]);
                  real_t* vbuff3_ptr=(vbuff[3].Dim(0)*vbuff[3].Dim(1)?vbuff[3][interac_idx]:trg_value[0]);
                  if(src_coord.Dim(1)){
                    {
                      real_t* shift=&intdata.coord_shift[int_id*3];
                      if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                        size_t vdim=src_coord.Dim(1);
                        Vector<real_t> new_coord(vdim, &buff[0], false);
                        assert(buff.Dim()>=vdim);
                        for(size_t j=0;j<vdim;j+=3){
                          for(size_t k=0;k<3;k++){
                            new_coord[j+k]=src_coord[0][j+k]+shift[k];
                          }
                        }
                        src_coord.ReInit(1, vdim, &new_coord[0], false);
                      }
                    }
                    setup_data.kernel->ker_poten(src_coord[0], src_coord.Dim(1)/3, vbuff2_ptr,
                                                 trg_coord[0], trg_coord.Dim(1)/3, vbuff3_ptr);
                  }
                  if(srf_coord.Dim(1)){
                    {
                      real_t* shift=&intdata.coord_shift[int_id*3];
                      if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                        size_t vdim=srf_coord.Dim(1);
                        Vector<real_t> new_coord(vdim, &buff[0], false);
                        assert(buff.Dim()>=vdim);
                        for(size_t j=0;j<vdim;j+=3){
                          for(size_t k=0;k<3;k++){
                            new_coord[j+k]=srf_coord[0][j+k]+shift[k];
                          }
                        }
                        srf_coord.ReInit(1, vdim, &new_coord[0], false);
                      }
                    }
                  }
                  interac_idx++;
                }
              }
            }
            if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){
              size_t interac_idx=0;
              for(size_t trg1=0;trg1<trg1_max;trg1++){
                size_t trg=trg0+trg1;
                for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                  size_t int_id=intdata.interac_dsp[trg]+i;
                  size_t scal_idx=intdata.scal_idx[int_id];
                  {
                    Matrix<real_t>& vec=vbuff[3];
                    Vector<real_t>& scal=intdata.scal[scal_idx*4+2];
                    size_t scal_dim=scal.Dim();
                    if(scal_dim){
                      size_t vdim=vec.Dim(1);
                      for(size_t j=0;j<vdim;j+=scal_dim){
                        for(size_t k=0;k<scal_dim;k++){
                          vec[interac_idx][j+k]*=scal[k];
                        }
                      }
                    }
                  }
                  interac_idx++;
                }
              }
              Matrix<real_t>::GEMM(vbuff[4],vbuff[3],intdata.M[2]);
              Matrix<real_t>::GEMM(vbuff[5],vbuff[4],intdata.M[3]);
              interac_idx=0;
              for(size_t trg1=0;trg1<trg1_max;trg1++){
                size_t trg=trg0+trg1;
                trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
                for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                  size_t int_id=intdata.interac_dsp[trg]+i;
                  size_t scal_idx=intdata.scal_idx[int_id];
                  {
                    Matrix<real_t>& vec=vbuff[5];
                    Vector<real_t>& scal=intdata.scal[scal_idx*4+3];
                    size_t scal_dim=scal.Dim();
                    if(scal_dim){
                      size_t vdim=vec.Dim(1);
                      for(size_t j=0;j<vdim;j+=scal_dim){
                        for(size_t k=0;k<scal_dim;k++){
                          vec[interac_idx][j+k]*=scal[k];
                        }
                      }
                    }
                  }
                  {
                    size_t vdim=vbuff[5].Dim(1);
                    assert(trg_value.Dim(1)==vdim);
                    for(size_t i=0;i<vdim;i++) trg_value[0][i]+=vbuff[5][interac_idx][i];
                  }
                  interac_idx++;
                }
              }
            }
            trg0+=trg1_max;
          }
        }
      }
    }
    Profile::Toc();
  }

  void EvalList(SetupData& setup_data){
    if(setup_data.interac_cnt.empty()){
      return;
    }
    Profile::Tic("Host2Device",false,25);
    char* buff = dev_buffer.data_ptr;
    char* precomp_data = &(*setup_data.precomp_data)[0];
    real_t* input_data = setup_data.input_data->data_ptr;
    real_t* output_data = setup_data.output_data->data_ptr;
    Profile::Toc();
    Profile::Tic("DeviceComp",false,20);
    int lock_idx=-1;
    int wait_lock_idx=-1;
    {
      size_t M_dim0 = setup_data.M_dim0;
      size_t M_dim1 = setup_data.M_dim1;
      Vector<size_t> interac_blk = setup_data.interac_blk;
      Vector<size_t> interac_cnt = setup_data.interac_cnt;
      Vector<size_t> interac_mat = setup_data.interac_mat;
      Vector<size_t>  input_perm = setup_data.input_perm;
      Vector<size_t> output_perm = setup_data.output_perm;
      {
        int omp_p=omp_get_max_threads();
        size_t interac_indx=0;
        size_t interac_blk_dsp=0;
        for(size_t k=0;k<interac_blk.Dim();k++){
          size_t vec_cnt=0;
          for(size_t j=interac_blk_dsp;j<interac_blk_dsp+interac_blk[k];j++) vec_cnt+=interac_cnt[j];
          if(vec_cnt==0){
            interac_blk_dsp += interac_blk[k];
            continue;
          }
          char* buff_in =buff;
          char* buff_out=buff+vec_cnt*M_dim0*sizeof(real_t);
#pragma omp parallel for
          for(int tid=0;tid<omp_p;tid++){
            size_t a=( tid   *vec_cnt)/omp_p;
            size_t b=((tid+1)*vec_cnt)/omp_p;
            for(size_t i=a;i<b;i++){
              const size_t*  perm=(size_t*)(precomp_data+input_perm[(interac_indx+i)*4+0]);
              const real_t*  scal=(real_t*)(precomp_data+input_perm[(interac_indx+i)*4+1]);
              const real_t* v_in =(real_t*)(  input_data+input_perm[(interac_indx+i)*4+3]);
              real_t*       v_out=(real_t*)(     buff_in+input_perm[(interac_indx+i)*4+2]);
              for(size_t j=0;j<M_dim0;j++ ){
                v_out[j]=v_in[perm[j]]*scal[j];
              }
            }
          }
          size_t vec_cnt0=0;
          for(size_t j=interac_blk_dsp;j<interac_blk_dsp+interac_blk[k];){
            size_t vec_cnt1=0;
            size_t interac_mat0=interac_mat[j];
            for(;j<interac_blk_dsp+interac_blk[k] && interac_mat[j]==interac_mat0;j++) vec_cnt1+=interac_cnt[j];
            Matrix<real_t> M(M_dim0, M_dim1, (real_t*)(precomp_data+interac_mat0), false);
#pragma omp parallel for
            for(int tid=0;tid<omp_p;tid++){
              size_t a=(vec_cnt1*(tid  ))/omp_p;
              size_t b=(vec_cnt1*(tid+1))/omp_p;
              Matrix<real_t> Ms(b-a, M_dim0, (real_t*)(buff_in +M_dim0*vec_cnt0*sizeof(real_t))+M_dim0*a, false);
              Matrix<real_t> Mt(b-a, M_dim1, (real_t*)(buff_out+M_dim1*vec_cnt0*sizeof(real_t))+M_dim1*a, false);
              Matrix<real_t>::GEMM(Mt,Ms,M);
            }
            vec_cnt0+=vec_cnt1;
          }
#pragma omp parallel for
          for(int tid=0;tid<omp_p;tid++){
            size_t a=( tid   *vec_cnt)/omp_p;
            size_t b=((tid+1)*vec_cnt)/omp_p;
            if(tid>      0 && a<vec_cnt){
              size_t out_ptr=output_perm[(interac_indx+a)*4+3];
              if(tid>      0) while(a<vec_cnt && out_ptr==output_perm[(interac_indx+a)*4+3]) a++;
            }
            if(tid<omp_p-1 && b<vec_cnt){
              size_t out_ptr=output_perm[(interac_indx+b)*4+3];
              if(tid<omp_p-1) while(b<vec_cnt && out_ptr==output_perm[(interac_indx+b)*4+3]) b++;
            }
            for(size_t i=a;i<b;i++){ // Compute permutations.
              const size_t*  perm=(size_t*)(precomp_data+output_perm[(interac_indx+i)*4+0]);
              const real_t*  scal=(real_t*)(precomp_data+output_perm[(interac_indx+i)*4+1]);
              const real_t* v_in =(real_t*)(    buff_out+output_perm[(interac_indx+i)*4+2]);
              real_t*       v_out=(real_t*)( output_data+output_perm[(interac_indx+i)*4+3]);
              for(size_t j=0;j<M_dim1;j++ ){
                v_out[j]+=v_in[perm[j]]*scal[j];
              }
            }
          }
          interac_indx+=vec_cnt;
          interac_blk_dsp+=interac_blk[k];
        }
      }
    }
    Profile::Toc();
  }

  void Source2Up(SetupData&  setup_data) {
    if(!multipole_order) return;
    EvalListPts(setup_data);
  }

  void Up2Up(SetupData& setup_data){
    if(!multipole_order) return;
    EvalList(setup_data);
  }

  void X_List(SetupData&  setup_data){
    if(!multipole_order) return;
    EvalListPts(setup_data);
  }

  void W_List(SetupData&  setup_data){
    if(!multipole_order) return;
    EvalListPts(setup_data);
  }

  void U_List(SetupData&  setup_data){
    EvalListPts(setup_data);
  }

  void V_List(SetupData&  setup_data){
    if(!multipole_order) return;
    int np=1;
    Profile::Tic("Host2Device",false,25);
    int level=setup_data.level;
    int dim0=setup_data.input_data->dim[0];
    int dim1=setup_data.input_data->dim[1];
    size_t buff_size=*((size_t*)&setup_data.vlist_data.buff_size);
    if(dev_buffer.Dim()<buff_size) dev_buffer.Resize(buff_size);
    char * buff=dev_buffer.data_ptr;
    VListData vlist_data=setup_data.vlist_data;
    real_t * input_data=setup_data.input_data->data_ptr;
    real_t * output_data=setup_data.output_data->data_ptr;
    Profile::Toc();
    buff_size     = vlist_data.buff_size;
    size_t m      = vlist_data.m;
    size_t n_blk0 = vlist_data.n_blk0;
    size_t n1 = m * 2;
    size_t n2 = n1 * n1;
    size_t n3_ = n2 * (n1 / 2 + 1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t M_dim = n3_;
    std::vector<real_t*> precomp_mat = vlist_data.precomp_mat;
    std::vector<std::vector<size_t> >  fft_vec = vlist_data.fft_vec;
    std::vector<std::vector<size_t> > ifft_vec = vlist_data.ifft_vec;
    std::vector<std::vector<real_t> >  fft_scl = vlist_data.fft_scl;
    std::vector<std::vector<real_t> > ifft_scl = vlist_data.ifft_scl;
    std::vector<std::vector<size_t> > interac_vec = vlist_data.interac_vec;
    std::vector<std::vector<size_t> > interac_dsp = vlist_data.interac_dsp;
    int omp_p=omp_get_max_threads();
    for(size_t blk0=0;blk0<n_blk0;blk0++){
      size_t n_in = fft_vec[blk0].size();
      size_t n_out=ifft_vec[blk0].size();
      size_t  input_dim=n_in *fftsize;
      size_t output_dim=n_out*fftsize;
      size_t buffer_dim=4*fftsize*omp_p;
      Vector<real_t> fft_in ( input_dim, (real_t*)buff,false);
      Vector<real_t> fft_out(output_dim, (real_t*)(buff+input_dim*sizeof(real_t)),false);
      Vector<real_t>  buffer(buffer_dim, (real_t*)(buff+(input_dim+output_dim)*sizeof(real_t)),false);
      Vector<real_t>  input_data_(dim0*dim1,input_data,false);
      FFT_UpEquiv(m, fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in, buffer);
      VListHadamard(M_dim, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
      Vector<real_t> output_data_(dim0*dim1, output_data, false);
      FFT_Check2Equiv(m, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_, buffer);
    }
  }


  void Down2Down(SetupData& setup_data){
    if(!multipole_order) return;
    EvalList(setup_data);
  }
  void Down2Target(SetupData&  setup_data){
    if(!multipole_order) return;
    EvalListPts(setup_data);
  }

  void UpwardPass() {
    int depth=0;
    std::vector<FMM_Node*>& nodes=GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMM_Node* n=nodes[i];
      if(n->depth>depth) depth=n->depth;
    }
    Profile::Tic("S2U",false,5);
    Source2Up(setup_data[MAX_DEPTH*6]);
    Profile::Toc();
    Profile::Tic("U2U",false,5);
    for(int i=depth-1; i>=0; i--){
      Up2Up(setup_data[i+MAX_DEPTH*7]);
    }
    Profile::Toc();
  }

  void DownwardPass() {
    Profile::Tic("Setup",true,3);
    std::vector<FMM_Node*> leaf_nodes;
    int depth=0;
    std::vector<FMM_Node*>& nodes=GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMM_Node* n=nodes[i];
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes.push_back(n);
      if(n->depth>depth) depth=n->depth;
    }
    Profile::Toc();
    Profile::Tic("X-List",false,5);
    X_List(setup_data[MAX_DEPTH*2]);
    Profile::Toc();
    Profile::Tic("W-List",false,5);
    W_List(setup_data[MAX_DEPTH*1]);
    Profile::Toc();
    Profile::Tic("U-List",false,5);
    U_List(setup_data[MAX_DEPTH*0]);
    Profile::Toc();
    Profile::Tic("V-List",false,5);
    V_List(setup_data[MAX_DEPTH*3]);
    Profile::Toc();
    Profile::Tic("D2D",false,5);
    for(size_t i=0; i<=depth; i++) {
      Down2Down(setup_data[i+MAX_DEPTH*4]);
    }
    Profile::Toc();
    Profile::Tic("D2T",false,5);
    Down2Target(setup_data[MAX_DEPTH*5]);
    Profile::Toc();
  }

public:
  void RunFMM() {
    Profile::Tic("RunFMM",true);
    {
      Profile::Tic("UpwardPass",false,2);
      UpwardPass();
      Profile::Toc();
      Profile::Tic("DownwardPass",true,2);
      DownwardPass();
      Profile::Toc();
    }
    Profile::Toc();
  }

  void CheckFMMOutput(std::string t_name){
    int np=omp_get_max_threads();
    int myrank=0, p=1;

    std::vector<real_t> src_coord;
    std::vector<real_t> src_value;
    FMM_Node* n=root_node;
    while(n!=NULL){
      if(n->IsLeaf() && !n->IsGhost()){
        pvfmm::Vector<real_t>& coord_vec=n->src_coord;
        pvfmm::Vector<real_t>& value_vec=n->src_value;
        for(size_t i=0;i<coord_vec.Dim();i++) src_coord.push_back(coord_vec[i]);
        for(size_t i=0;i<value_vec.Dim();i++) src_value.push_back(value_vec[i]);
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    long long src_cnt=src_coord.size()/3;
    long long val_cnt=src_value.size();
    if(src_cnt==0) return;
    int dof=val_cnt/src_cnt/kernel->ker_dim[0];
    int trg_dof=dof*kernel->ker_dim[1];
    std::vector<real_t> trg_coord;
    std::vector<real_t> trg_poten_fmm;
    long long trg_iter=0;
    size_t step_size=1+src_cnt*src_cnt*1e-9/p;
    n=root_node;
    while(n!=NULL){
      if(n->IsLeaf() && !n->IsGhost()){
        pvfmm::Vector<real_t>& coord_vec=n->trg_coord;
        pvfmm::Vector<real_t>& poten_vec=n->trg_value;
        for(size_t i=0;i<coord_vec.Dim()/3          ;i++){
          if(trg_iter%step_size==0){
            for(int j=0;j<3        ;j++) trg_coord    .push_back(coord_vec[i*3        +j]);
            for(int j=0;j<trg_dof  ;j++) trg_poten_fmm.push_back(poten_vec[i*trg_dof  +j]);
          }
          trg_iter++;
        }
      }
      n=static_cast<FMM_Node*>(PreorderNxt(n));
    }
    int trg_cnt=trg_coord.size()/3;
    if(trg_cnt==0) return;
    std::vector<real_t> trg_poten_dir(trg_cnt*trg_dof ,0);
    pvfmm::Profile::Tic("N-Body Direct",false,1);
    #pragma omp parallel for
    for(int i=0;i<np;i++){
      size_t a=(i*trg_cnt)/np;
      size_t b=((i+1)*trg_cnt)/np;
      kernel->ker_poten(&src_coord[0], src_cnt, &src_value[0], &trg_coord[a*3], b-a, &trg_poten_dir[a*trg_dof  ]);
    }
    pvfmm::Profile::Toc();
    {
      real_t max_=0;
      real_t max_err=0;
      for(size_t i=0;i<trg_poten_fmm.size();i++){
        real_t err=fabs(trg_poten_dir[i]-trg_poten_fmm[i]);
        real_t max=fabs(trg_poten_dir[i]);
        if(err>max_err) max_err=err;
        if(max>max_) max_=max;
      }
      if(!myrank){
        std::cout<<"Error      : "<<std::scientific<<max_err/max_<<'\n';
      }
    }
  }

};

}//end namespace

#undef fft_plan_many_dft_r2c
#undef fft_plan_many_dft_c2r
#undef fft_execute_dft_r2c
#undef fft_execute_dft_c2r
#undef fft_destroy_plan

#endif //_PVFMM_FMM_TREE_HPP_
