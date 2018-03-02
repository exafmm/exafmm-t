#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_
#include "pvfmm.h"
#include "kernel.hpp"
#include "interac_list.hpp"
#include "geometry.h"
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

  PrecompMat(InteracList* interacList_, int multipole_order_, const Kernel* kernel_) {
    multipole_order = multipole_order_;
    kernel = kernel_;
    interacList = interacList_;
    mat.resize(Type_Count);
    for(size_t i=0;i<mat.size();i++)
      mat[i].resize(500);

    perm.resize(Type_Count);
    for(size_t i=0;i<Type_Count;i++){
      perm[i].resize(Perm_Count);
    }
    perm_r.resize(256*Type_Count);
    perm_c.resize(256*Type_Count);
    for(size_t i=0;i<perm_r.size();i++){
      perm_r[i].resize(500);
      perm_c[i].resize(500);
    }
  }

  Permutation<real_t>& getPerm_R(int l, Mat_Type type, size_t indx){
    int level=l+128;
    assert(level*Type_Count+type<int(perm_r.size()));
    if(indx>=perm_r[level*Type_Count+type].size()){
      perm_r[level*Type_Count+type].resize(indx+1);
    }
    return perm_r[level*Type_Count+type][indx];
  }

  Permutation<real_t>& getPerm_C(int l, Mat_Type type, size_t indx){
    int level=l+128;
    assert(level*Type_Count+type<int(perm_c.size()));
    if(indx>=perm_c[level*Type_Count+type].size()){
      perm_c[level*Type_Count+type].resize(indx+1);
    }
    return perm_c[level*Type_Count+type][indx];
  }
  
  // This is only related to M2M and L2L operator
  Permutation<real_t>& Perm_R(int l, Mat_Type type, size_t indx){
    size_t indx0 = interacList->interac_class[type][indx];                     // indx0: class coord index
    Matrix     <real_t>& M0      = mat[type][indx0];         // class coord matrix
    Permutation<real_t>& row_perm = getPerm_R(l, type, indx );    // mat->perm_r[(l+128)*16+type][indx]
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return row_perm;             // if mat hasn't been computed, then return
    if(row_perm.Dim()==0){                                        // if this perm_r entry hasn't been computed
      std::vector<Perm_Type> p_list = interacList->perm_list[type][indx];      // get perm_list of current rel_coord
      for(int i=0;i<l;i++) p_list.push_back(Scaling);             // push back Scaling operation l times
      Permutation<real_t> row_perm_=Permutation<real_t>(M0.Dim(0));  // init row_perm to be size npts*src_dim
      for(int i=0;i<C_Perm;i++){                                  // loop over permutation types
	Permutation<real_t>& pr = perm[type][R_Perm + i];      // grab the handle of its mat->perm entry
	if(!pr.Dim()) row_perm_ = Permutation<real_t>(0);           // if PrecompPerm never called for this type and entry: this entry does not need permutation so set it empty
      }
      if(row_perm_.Dim()>0)                                       // if this type & entry needs permutation
	for(int i=p_list.size()-1; i>=0; i--){                    // loop over the operations of perm_list from end to begin
	  //assert(type!=V_Type);
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
	  //assert(type!=V_Type);
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

  Permutation<real_t>& PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<real_t>& P_ = perm[type][perm_indx];
    if(P_.Dim()!=0) return P_;
    size_t m=multipole_order;         //
    size_t p_indx=perm_indx % C_Perm;
    Permutation<real_t> P;
    switch (type) {
    case M2M_Type: {
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
    case L2L_Type: {
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
  
  inline uintptr_t align_ptr(uintptr_t ptr){
    static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
    static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
    return ((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
  }

  size_t CompactData(int level, Mat_Type type, std::vector<char>& comp_data, size_t offset=0){
    struct HeaderData{
      size_t total_size;
      size_t      level;
      size_t   mat_cnt ;
      size_t  max_depth;
    };
    if(comp_data.size()>offset){
      char* indx_ptr=&comp_data[0]+offset;
      HeaderData& header=*(HeaderData*)indx_ptr; indx_ptr+=sizeof(HeaderData);
      if(level==int(header.level)){
	offset+=header.total_size;
	return offset;
      }
    }
    std::vector<Matrix<real_t> >& mat_ = mat[type];
    size_t mat_cnt=mat_.size();
    size_t indx_size=0;
    size_t mem_size=0;
    int omp_p=omp_get_max_threads();
    size_t l0=0;
    size_t l1=128;
    {
      indx_size+=sizeof(HeaderData);
      indx_size+=mat_cnt*(1+(2+2)*(l1-l0))*sizeof(size_t);
      indx_size=align_ptr(indx_size);

      for(size_t j=0;j<mat_cnt;j++){
	Matrix     <real_t>& M = mat[type][j];
	if(M.Dim(0)>0 && M.Dim(1)>0){
	  mem_size+=M.Dim(0)*M.Dim(1)*sizeof(real_t); mem_size=align_ptr(mem_size);
	}

	for(size_t l=l0;l<l1;l++){
	  Permutation<real_t>& Pr=getPerm_R(l,type,j);
	  Permutation<real_t>& Pc=getPerm_C(l,type,j);
	  if(Pr.Dim()>0){
	    mem_size+=Pr.Dim()*sizeof(size_t); mem_size=align_ptr(mem_size);
	    mem_size+=Pr.Dim()*sizeof(real_t); mem_size=align_ptr(mem_size);
	  }
	  if(Pc.Dim()>0){
	    mem_size+=Pc.Dim()*sizeof(size_t); mem_size=align_ptr(mem_size);
	    mem_size+=Pc.Dim()*sizeof(real_t); mem_size=align_ptr(mem_size);
	  }
	}
      }
    }
    if(comp_data.size()<offset+indx_size+mem_size){
      std::vector<char> old_data;
      if(offset>0) old_data=comp_data;
      comp_data.resize(offset+indx_size+mem_size);
      if(offset>0){
#pragma omp parallel for
	for(int tid=0;tid<omp_p;tid++){
	  size_t a=(offset*(tid+0))/omp_p;
	  size_t b=(offset*(tid+1))/omp_p;
	  memcpy(&comp_data[0]+a, &old_data[0]+a, b-a);
	}
      }
    }
    {
      char* indx_ptr=&comp_data[0]+offset;
      HeaderData& header=*(HeaderData*)indx_ptr; indx_ptr+=sizeof(HeaderData);
      Matrix<size_t> offset_indx(mat_cnt,1+(2+2)*(l1-l0), (size_t*)indx_ptr, false);
      header.total_size=indx_size+mem_size;
      header.     level=level             ;
      header.  mat_cnt = mat_cnt          ;
      header. max_depth=l1-l0             ;
      size_t data_offset=offset+indx_size;
      for(size_t j=0;j<mat_cnt;j++){
	Matrix     <real_t>& M = mat[type][j];
	offset_indx[j][0]=data_offset; indx_ptr+=sizeof(size_t);
	data_offset+=M.Dim(0)*M.Dim(1)*sizeof(real_t); mem_size=align_ptr(mem_size);
	for(size_t l=l0;l<l1;l++){
	  Permutation<real_t>& Pr=getPerm_R(l,type,j);
	  offset_indx[j][1+4*(l-l0)+0]=data_offset;
	  data_offset+=Pr.Dim()*sizeof(size_t); mem_size=align_ptr(mem_size);
	  offset_indx[j][1+4*(l-l0)+1]=data_offset;
	  data_offset+=Pr.Dim()*sizeof(real_t); mem_size=align_ptr(mem_size);
	  Permutation<real_t>& Pc=getPerm_C(l,type,j);
	  offset_indx[j][1+4*(l-l0)+2]=data_offset;
	  data_offset+=Pc.Dim()*sizeof(size_t); mem_size=align_ptr(mem_size);
	  offset_indx[j][1+4*(l-l0)+3]=data_offset;
	  data_offset+=Pc.Dim()*sizeof(real_t); mem_size=align_ptr(mem_size);
	}
      }
    }
#pragma omp parallel for
    for(int tid=0;tid<omp_p;tid++){
      char* indx_ptr=&comp_data[0]+offset;
      indx_ptr+=sizeof(HeaderData);
      Matrix<size_t> offset_indx(mat_cnt,1+(2+2)*(l1-l0), (size_t*)indx_ptr, false);
      for(size_t j=0;j<mat_cnt;j++){
	Matrix     <real_t>& M = mat[type][j];
	if(M.Dim(0)>0 && M.Dim(1)>0){
	  size_t a=(M.Dim(0)*M.Dim(1)* tid   )/omp_p;
	  size_t b=(M.Dim(0)*M.Dim(1)*(tid+1))/omp_p;
	  memcpy(&comp_data[0]+offset_indx[j][0]+a*sizeof(real_t), &M[0][a], (b-a)*sizeof(real_t));
	}
	for(size_t l=l0;l<l1;l++){
	  Permutation<real_t>& Pr=getPerm_R(l,type,j);
	  Permutation<real_t>& Pc=getPerm_C(l,type,j);
	  if(Pr.Dim()>0){
	    size_t a=(Pr.Dim()* tid   )/omp_p;
	    size_t b=(Pr.Dim()*(tid+1))/omp_p;
	    memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+0]+a*sizeof(size_t), &Pr.perm[a], (b-a)*sizeof(size_t));
	    memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+1]+a*sizeof(real_t), &Pr.scal[a], (b-a)*sizeof(real_t));
	  }
	  if(Pc.Dim()>0){
	    size_t a=(Pc.Dim()* tid   )/omp_p;
	    size_t b=(Pc.Dim()*(tid+1))/omp_p;
	    memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+2]+a*sizeof(size_t), &Pc.perm[a], (b-a)*sizeof(size_t));
	    memcpy(&comp_data[0]+offset_indx[j][1+4*(l-l0)+3]+a*sizeof(real_t), &Pc.scal[a], (b-a)*sizeof(real_t));
	  }
	}
      }
    }
    return offset+indx_size+mem_size;
  }

  void Save2File(const char* fname, bool replace=false){
    FILE* f=fopen(fname,"r");
    if(f!=NULL) {
      fclose(f);
      if(!replace) return;
    }
    f=fopen(fname,"wb");
    if(f==NULL) return;
    int tmp;
    tmp=sizeof(real_t);
    fwrite(&tmp,sizeof(int),1,f);
    tmp=1;
    fwrite(&tmp,sizeof(int),1,f);
    for(size_t i=0;i<mat.size();i++){
      int n=mat[i].size();
      fwrite(&n,sizeof(int),1,f);
      for(int j=0;j<n;j++){
	Matrix<real_t>& M=mat[i][j];
	int n1=M.Dim(0);
	fwrite(&n1,sizeof(int),1,f);
	int n2=M.Dim(1);
	fwrite(&n2,sizeof(int),1,f);
	if(n1*n2>0)
	  fwrite(&M[0][0],sizeof(real_t),n1*n2,f);
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
    Profile::Tic("LoadMatrices",true,3);
    Profile::Tic("ReadFile",true,4);
    size_t f_size=0;
    char* f_data=NULL;

    FILE* f=fopen(fname,"rb");
    if(f==NULL){
      f_size=0;
      std::cout << "No existing precomputation matrix file" << std::endl;
    }else{
      struct stat fileStat;
      if(stat(fname,&fileStat) < 0) f_size=0;
      else f_size=fileStat.st_size;
    }
    if(f_size>0){
      f_data= new char [f_size];
      fseek (f, 0, SEEK_SET);
      MY_FREAD(f_data,sizeof(char),f_size,f);
      fclose(f);
    }

    Profile::Toc();
    Profile::Tic("Broadcast",true,4);
    if(f_size==0){
      Profile::Toc();
      Profile::Toc();
      return;
    }
    if(f_data==NULL) f_data=new char [f_size];
    char* f_ptr=f_data;
    int max_send_size=1000000000;
    while(f_size>0){
      if(f_size>(size_t)max_send_size){
	f_size-=max_send_size;
	f_ptr+=max_send_size;
      }else{
	f_size=0;
      }
    }
    f_ptr=f_data;
    {
      int tmp;
      tmp=*(int*)f_ptr; f_ptr+=sizeof(int);
      assert(tmp==sizeof(real_t));
      tmp=*(int*)f_ptr; f_ptr+=sizeof(int);
      size_t mat_size=(size_t)Type_Count*1;
      if(mat.size()<mat_size){
	mat.resize(mat_size);
      }
      for(size_t i=0;i<mat_size;i++){
	int n;
	n=*(int*)f_ptr; f_ptr+=sizeof(int);
	if(mat[i].size()<(size_t)n)
	  mat[i].resize(n);
	for(int j=0;j<n;j++){
	  Matrix<real_t>& M=mat[i][j];
	  int n1;
	  n1=*(int*)f_ptr; f_ptr+=sizeof(int);
	  int n2;
	  n2=*(int*)f_ptr; f_ptr+=sizeof(int);
	  if(n1*n2>0){
	    M.Resize(n1,n2);
	    memcpy(&M[0][0], f_ptr, sizeof(real_t)*n1*n2); f_ptr+=sizeof(real_t)*n1*n2;
	  }
	}
      }
      perm_r.resize(256*Type_Count);
      perm_c.resize(256*Type_Count);
      for(size_t i=0;i<perm_r.size();i++){
	perm_r[i].resize(500);
	perm_c[i].resize(500);
      }
    }
    delete[] f_data;
    Profile::Toc();
    Profile::Toc();
  }

#undef MY_FREAD
};

}//end namespace

#endif //_PrecompMAT_HPP_
