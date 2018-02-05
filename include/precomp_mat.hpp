#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_

namespace pvfmm{

typedef enum{
  UC2UE0_Type= 0,
  UC2UE1_Type= 1,
  DC2DE0_Type= 2,
  DC2DE1_Type= 3,
  S2U_Type  = 4,
  U2U_Type  = 5,
  D2D_Type  = 6,
  D2T_Type  = 7,
  U0_Type   = 8,
  U1_Type   = 9,
  U2_Type   =10,
  V_Type    =11,
  W_Type    =12,
  X_Type    =13,
  V1_Type   =14,
  Type_Count=15
} Mat_Type;

typedef enum{
  Scaling = 0,
  ReflecX = 1,
  ReflecY = 2,
  ReflecZ = 3,
  SwapXY  = 4,
  SwapXZ  = 5,
  R_Perm = 0,
  C_Perm = 6,
  Perm_Count=12
} Perm_Type;

class PrecompMat{
public:
  std::vector<std::vector<Matrix<real_t> > > mat;
  std::vector<std::vector<Permutation<real_t> > > perm;

  PrecompMat() {
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

  Permutation<real_t>& Perm_R(int l, Mat_Type type, size_t indx){
    int level=l+128;
    assert(level*Type_Count+type<perm_r.size());
    if(indx>=perm_r[level*Type_Count+type].size()){
      perm_r[level*Type_Count+type].resize(indx+1);
    }
    return perm_r[level*Type_Count+type][indx];
  }

  Permutation<real_t>& Perm_C(int l, Mat_Type type, size_t indx){
    int level=l+128;
    assert(level*Type_Count+type<perm_c.size());
    if(indx>=perm_c[level*Type_Count+type].size()){
      perm_c[level*Type_Count+type].resize(indx+1);
    }
    return perm_c[level*Type_Count+type][indx];
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
      if(level==header.level){
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
	  Permutation<real_t>& Pr=Perm_R(l,type,j);
	  Permutation<real_t>& Pc=Perm_C(l,type,j);
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
	  Permutation<real_t>& Pr=Perm_R(l,type,j);
	  offset_indx[j][1+4*(l-l0)+0]=data_offset;
	  data_offset+=Pr.Dim()*sizeof(size_t); mem_size=align_ptr(mem_size);
	  offset_indx[j][1+4*(l-l0)+1]=data_offset;
	  data_offset+=Pr.Dim()*sizeof(real_t); mem_size=align_ptr(mem_size);
	  Permutation<real_t>& Pc=Perm_C(l,type,j);
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
	  Permutation<real_t>& Pr=Perm_R(l,type,j);
	  Permutation<real_t>& Pc=Perm_C(l,type,j);
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

 private:
  std::vector<std::vector<Permutation<real_t> > > perm_r;
  std::vector<std::vector<Permutation<real_t> > > perm_c;
};

}//end namespace

#endif //_PrecompMAT_HPP_
