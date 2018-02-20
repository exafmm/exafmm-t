#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_

extern "C" {
  void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA, float* A,
	      int* LDA, float* B, int* LDB, float* BETA, float* C, int* LDC);
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
	      int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  void sgesvd_(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
	       float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO);
  void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
	       double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
}

namespace pvfmm{

  template <class T>
  void tpinv(T* M, int n1, int n2, T eps, T* M_){
    if(n1*n2==0) return;
    int m = n2;
    int n = n1;
    int k = (m<n?m:n);
    int err;
    T *tU, *tS, *tVT, *wsbuf;
    err = posix_memalign((void**)&tU, MEM_ALIGN, m*k*sizeof(T));
    err = posix_memalign((void**)&tS, MEM_ALIGN, k*sizeof(T));
    err = posix_memalign((void**)&tVT, MEM_ALIGN, k*n*sizeof(T));
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);
    err = posix_memalign((void**)&wsbuf, MEM_ALIGN, wssize*sizeof(T));
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, wsbuf, &wssize, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, wsbuf, &wssize, &INFO);
#endif
    if(INFO!=0)
      std::cout<<INFO<<'\n';
    assert(INFO==0);
    free(wsbuf);
    T eps_=tS[0]*eps;
    for(int i=0;i<k;i++)
      if(tS[i]<eps_)
        tS[i]=0;
      else
        tS[i]=1.0/tS[i];
    for(int i=0;i<m;i++){
      for(int j=0;j<k;j++){
        tU[i+j*m]*=tS[j];
      }
    }
    char TransA = 'T', TransB = 'T';
    real_t zero = 0, one = 1;
#if FLOAT
    sgemm_(&TransA,&TransB,&n,&m,&k,&one,&tVT[0],&k,&tU[0],&m,&zero,M_,&n);
#else
    dgemm_(&TransA,&TransB,&n,&m,&k,&one,&tVT[0],&k,&tU[0],&m,&zero,M_,&n);
#endif
    free(tU);
    free(tS);
    free(tVT);
  }

template <class T>
class Permutation;

template <class T>
class Matrix{

public:
  T* data_ptr;
  size_t dim[2];

  private:

  bool own_data;

  static inline void gemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc){
    if((TransA=='N' || TransA=='n') && (TransB=='N' || TransB=='n')){
      for(size_t n=0;n<N;n++){
        for(size_t m=0;m<M;m++){
            T AxB=0;
            for(size_t k=0;k<K;k++){
              AxB+=A[m+lda*k]*B[k+ldb*n];
            }
            C[m+ldc*n]=alpha*AxB+(beta==0?0:beta*C[m+ldc*n]);
        }
      }
    }else if(TransA=='N' || TransA=='n'){
      for(size_t n=0;n<N;n++){
        for(size_t m=0;m<M;m++){
            T AxB=0;
            for(size_t k=0;k<K;k++){
              AxB+=A[m+lda*k]*B[n+ldb*k];
            }
            C[m+ldc*n]=alpha*AxB+(beta==0?0:beta*C[m+ldc*n]);
        }
      }
    }else if(TransB=='N' || TransB=='n'){
      for(size_t n=0;n<N;n++){
        for(size_t m=0;m<M;m++){
            T AxB=0;
            for(size_t k=0;k<K;k++){
              AxB+=A[k+lda*m]*B[k+ldb*n];
            }
            C[m+ldc*n]=alpha*AxB+(beta==0?0:beta*C[m+ldc*n]);
        }
      }
    }else{
      for(size_t n=0;n<N;n++){
        for(size_t m=0;m<M;m++){
            T AxB=0;
            for(size_t k=0;k<K;k++){
              AxB+=A[k+lda*m]*B[n+ldb*k];
            }
            C[m+ldc*n]=alpha*AxB+(beta==0?0:beta*C[m+ldc*n]);
        }
      }
    }
  }

  public:

  Matrix(){
    dim[0]=0;
    dim[1]=0;
    own_data=true;
    data_ptr=NULL;
  }

  Matrix(size_t dim1, size_t dim2, T* data_=NULL, bool own_data_=true) {
    dim[0]=dim1;
    dim[1]=dim2;
    own_data=own_data_;
    if(own_data){
      if(dim[0]*dim[1]>0){
        int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim[0]*dim[1]*sizeof(T));
	if(data_!=NULL) memcpy(data_ptr,data_,dim[0]*dim[1]*sizeof(T));
      }else data_ptr=NULL;
    }else
      data_ptr=data_;
  }

  Matrix(const Matrix<T>& M){
    dim[0]=M.dim[0];
    dim[1]=M.dim[1];

    own_data=true;
    if(dim[0]*dim[1]>0){
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim[0]*dim[1]*sizeof(T));
      memcpy(data_ptr,M.data_ptr,dim[0]*dim[1]*sizeof(T));
    }else
      data_ptr=NULL;
  }

  ~Matrix(){
    if(own_data){
      if(data_ptr!=NULL){
	free(data_ptr);
      }
    }
    data_ptr=NULL;
    dim[0]=0;
    dim[1]=0;
  }

  void Swap(Matrix<T>& M){
    size_t dim_[2]={dim[0],dim[1]};
    T* data_ptr_=data_ptr;
    bool own_data_=own_data;

    dim[0]=M.dim[0];
    dim[1]=M.dim[1];
    data_ptr=M.data_ptr;
    own_data=M.own_data;

    M.dim[0]=dim_[0];
    M.dim[1]=dim_[1];
    M.data_ptr=data_ptr_;
    M.own_data=own_data_;
  }

  void ReInit(size_t dim1, size_t dim2, T* data_=NULL, bool own_data_=true){
    if(own_data_ && own_data && dim[0]*dim[1]>=dim1*dim2){
      dim[0]=dim1; dim[1]=dim2;
      if(data_) memcpy(data_ptr,data_,dim[0]*dim[1]*sizeof(T));
    }else{
      Matrix<T> tmp(dim1,dim2,data_,own_data_);
      this->Swap(tmp);
    }
  }

  void Write(const char* fname){
    FILE* f1=fopen(fname,"wb+");
    if(f1==NULL){
      std::cout<<"Unable to open file for writing:"<<fname<<'\n';
      return;
    }
    uint32_t dim_[2]={(uint32_t)dim[0],(uint32_t)dim[1]};
    fwrite(dim_,sizeof(uint32_t),2,f1);
    fwrite(data_ptr,sizeof(T),dim[0]*dim[1],f1);
    fclose(f1);
  }

  void Read(const char* fname){
    FILE* f1=fopen(fname,"r");
    if(f1==NULL){
      std::cout<<"Unable to open file for reading:"<<fname<<'\n';
      return;
    }
    uint32_t dim_[2];
    size_t readlen=fread (dim_, sizeof(uint32_t), 2, f1);
    assert(readlen==2);

    ReInit(dim_[0],dim_[1]);
    readlen=fread(data_ptr,sizeof(T),dim[0]*dim[1],f1);
    assert(readlen==dim[0]*dim[1]);
    fclose(f1);
  }

  size_t Dim(size_t i) const{
    return dim[i];
  }

  void Resize(size_t i, size_t j){
    if(dim[0]*dim[1]>=i*j){
      dim[0]=i; dim[1]=j;
    }else ReInit(i,j);
  }

  void SetZero(){
    if(dim[0]*dim[1])
      memset(data_ptr,0,dim[0]*dim[1]*sizeof(T));
  }

  Matrix<T>& operator=(const Matrix<T>& M){
    if(this!=&M){
      if(dim[0]*dim[1]<M.dim[0]*M.dim[1]){
	ReInit(M.dim[0],M.dim[1]);
      }
      dim[0]=M.dim[0]; dim[1]=M.dim[1];
      memcpy(data_ptr,M.data_ptr,dim[0]*dim[1]*sizeof(T));
    }
    return *this;
  }

  Matrix<T>& operator+=(const Matrix<T>& M){
    assert(M.Dim(0)==Dim(0) && M.Dim(1)==Dim(1));
    Profile::Add_FLOP(dim[0]*dim[1]);

    for(size_t i=0;i<M.Dim(0)*M.Dim(1);i++)
      data_ptr[i]+=M.data_ptr[i];
    return *this;
  }

  Matrix<T>& operator-=(const Matrix<T>& M){
    assert(M.Dim(0)==Dim(0) && M.Dim(1)==Dim(1));
    Profile::Add_FLOP(dim[0]*dim[1]);

    for(size_t i=0;i<M.Dim(0)*M.Dim(1);i++)
      data_ptr[i]-=M.data_ptr[i];
    return *this;
  }

  Matrix<T> operator+(const Matrix<T>& M2){
    Matrix<T>& M1=*this;
    assert(M2.Dim(0)==M1.Dim(0) && M2.Dim(1)==M1.Dim(1));
    Profile::Add_FLOP(dim[0]*dim[1]);
    Matrix<T> M_r(M1.Dim(0),M1.Dim(1),NULL);
    for(size_t i=0;i<M1.Dim(0)*M1.Dim(1);i++)
      M_r[0][i]=M1[0][i]+M2[0][i];
    return M_r;
  }

  Matrix<T> operator-(const Matrix<T>& M2){
    Matrix<T>& M1=*this;
    assert(M2.Dim(0)==M1.Dim(0) && M2.Dim(1)==M1.Dim(1));
    Profile::Add_FLOP(dim[0]*dim[1]);
    Matrix<T> M_r(M1.Dim(0),M1.Dim(1),NULL);
    for(size_t i=0;i<M1.Dim(0)*M1.Dim(1);i++)
      M_r[0][i]=M1[0][i]-M2[0][i];
    return M_r;
  }

  inline T& operator()(size_t i,size_t j) const{
    assert(i<dim[0] && j<dim[1]);
    return data_ptr[i*dim[1]+j];
  }

  inline T* operator[](size_t i) const{
    assert(i<dim[0]);
    return &data_ptr[i*dim[1]];
  }

  Matrix<T> operator*(const Matrix<T>& M){
    assert(dim[1]==M.dim[0]);
    Profile::Add_FLOP(2*(((long long)dim[0])*dim[1])*M.dim[1]);
    Matrix<T> M_r(dim[0],M.dim[1],NULL);
    if(M.Dim(0)*M.Dim(1)==0 || this->Dim(0)*this->Dim(1)==0) return M_r;
    gemm('N','N',M.dim[1],dim[0],dim[1],
		 1.0,M.data_ptr,M.dim[1],data_ptr,dim[1],0.0,M_r.data_ptr,M_r.dim[1]);
    return M_r;
  }

  static void GEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0){
    if(A.Dim(0)*A.Dim(1)==0 || B.Dim(0)*B.Dim(1)==0) return;
    assert(A.dim[1]==B.dim[0]);
    assert(M_r.dim[0]==A.dim[0]);
    assert(M_r.dim[1]==B.dim[1]);
    gemm('N','N',B.dim[1],A.dim[0],A.dim[1],
		 1.0,B.data_ptr,B.dim[1],A.data_ptr,A.dim[1],beta,M_r.data_ptr,M_r.dim[1]);
  }

#define myswap(t,a,b) {t c=a;a=b;b=c;}

  void RowPerm(const Permutation<T>& P){
    Matrix<T>& M=*this;
    if(P.Dim()==0) return;
    assert(M.Dim(0)==P.Dim());
    size_t d0=M.Dim(0);
    size_t d1=M.Dim(1);

#pragma omp parallel for
    for(size_t i=0;i<d0;i++){
    T* M_=M[i];
    const T s=P.scal[i];
    for(size_t j=0;j<d1;j++) M_[j]*=s;
  }

    Permutation<T> P_=P;
    for(size_t i=0;i<d0;i++)
      while(P_.perm[i]!=i){
    size_t a=P_.perm[i];
    size_t b=i;
    T* M_a=M[a];
    T* M_b=M[b];
    myswap(size_t,P_.perm[a],P_.perm[b]);
    for(size_t j=0;j<d1;j++)
      myswap(T,M_a[j],M_b[j]);
  }
  }

  void ColPerm(const Permutation<T>& P){
    Matrix<T>& M=*this;
    if(P.Dim()==0) return;
    assert(M.Dim(1)==P.Dim());
    size_t d0=M.Dim(0);
    size_t d1=M.Dim(1);

    int omp_p=omp_get_max_threads();
    Matrix<T> M_buff(omp_p,d1);

    const size_t* perm_=&(P.perm[0]);
    const T* scal_=&(P.scal[0]);
#pragma omp parallel for
    for(size_t i=0;i<d0;i++){
      int pid=omp_get_thread_num();
      T* buff=&M_buff[pid][0];
      T* M_=M[i];
      for(size_t j=0;j<d1;j++)
	buff[j]=M_[j];
      for(size_t j=0;j<d1;j++){
	M_[j]=buff[perm_[j]]*scal_[j];
      }
    }
  }
#undef myswap

#define B1 128
#define B2 32

  Matrix<T> Transpose(){
    Matrix<T>& M=*this;
    size_t d0=M.dim[0];
    size_t d1=M.dim[1];
    Matrix<T> M_r(d1,d0,NULL);

    const size_t blk0=((d0+B1-1)/B1);
    const size_t blk1=((d1+B1-1)/B1);
    const size_t blks=blk0*blk1;
    for(size_t k=0;k<blks;k++){
      size_t i=(k%blk0)*B1;
      size_t j=(k/blk0)*B1;
      size_t d0_=i+B1; if(d0_>=d0) d0_=d0;
      size_t d1_=j+B1; if(d1_>=d1) d1_=d1;
      for(size_t ii=i;ii<d0_;ii+=B2)
	for(size_t jj=j;jj<d1_;jj+=B2){
	  size_t d0__=ii+B2; if(d0__>=d0) d0__=d0;
	  size_t d1__=jj+B2; if(d1__>=d1) d1__=d1;
	  for(size_t iii=ii;iii<d0__;iii++)
	    for(size_t jjj=jj;jjj<d1__;jjj++){
	      M_r[jjj][iii]=M[iii][jjj];
	    }
	}
    }
    return M_r;
  }

  void Transpose(Matrix<T>& M_r, const Matrix<T>& M){
    size_t d0=M.dim[0];
    size_t d1=M.dim[1];
    M_r.Resize(d1, d0);

    const size_t blk0=((d0+B1-1)/B1);
    const size_t blk1=((d1+B1-1)/B1);
    const size_t blks=blk0*blk1;
#pragma omp parallel for
    for(size_t k=0;k<blks;k++){
      size_t i=(k%blk0)*B1;
      size_t j=(k/blk0)*B1;
      size_t d0_=i+B1; if(d0_>=d0) d0_=d0;
      size_t d1_=j+B1; if(d1_>=d1) d1_=d1;
      for(size_t ii=i;ii<d0_;ii+=B2)
	for(size_t jj=j;jj<d1_;jj+=B2){
	  size_t d0__=ii+B2; if(d0__>=d0) d0__=d0;
	  size_t d1__=jj+B2; if(d1__>=d1) d1__=d1;
	  for(size_t iii=ii;iii<d0__;iii++)
	    for(size_t jjj=jj;jjj<d1__;jjj++){
	      M_r[jjj][iii]=M[iii][jjj];
	    }
	}
    }
  }
#undef B2
#undef B1

  #define U(i,j) U_[(i)*dim[0]+(j)]
  #define S(i,j) S_[(i)*dim[1]+(j)]
  #define V(i,j) V_[(i)*dim[1]+(j)]

  static void GivensL(T* S_, const size_t dim[2], size_t m, T a, T b){
    T r=sqrtf(a*a+b*b);
    T c=a/r;
    T s=-b/r;
#pragma omp parallel for
    for(size_t i=0;i<dim[1];i++){
      T S0=S(m+0,i);
      T S1=S(m+1,i);
      S(m  ,i)+=S0*(c-1);
      S(m  ,i)+=S1*(-s );
      S(m+1,i)+=S0*( s );
      S(m+1,i)+=S1*(c-1);
    }
  }

  static void GivensR(T* S_, const size_t dim[2], size_t m, T a, T b){
    T r=sqrtf(a*a+b*b);
    T c=a/r;
  T s=-b/r;
#pragma omp parallel for
    for(size_t i=0;i<dim[0];i++){
      T S0=S(i,m+0);
      T S1=S(i,m+1);
      S(i,m  )+=S0*(c-1);
      S(i,m  )+=S1*(-s );
      S(i,m+1)+=S0*( s );
      S(i,m+1)+=S1*(c-1);
    }
  }

  #undef U
  #undef S
  #undef V


  void SVD(Matrix<T>& tU, Matrix<T>& tS, Matrix<T>& tVT){
    pvfmm::Matrix<T>& M=*this;
    pvfmm::Matrix<T> M_=M;
    int n=M.Dim(0);
    int m=M.Dim(1);
    int k = (m<n?m:n);
    tU.Resize(n,k); tU.SetZero();
    tS.Resize(k,k); tS.SetZero();
    tVT.Resize(k,m); tVT.SetZero();
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);
    T* wsbuf;
    int err = posix_memalign((void**)&wsbuf, MEM_ALIGN, wssize*sizeof(T));
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf, &wssize, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf, &wssize, &INFO);
#endif
    //svd(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf, &wssize, &INFO);
    free(wsbuf);
    if(INFO!=0) std::cout<<INFO<<'\n';
    assert(INFO==0);
    for(size_t i=1;i<k;i++){
    tS[i][i]=tS[0][i];
    tS[0][i]=0;
  }
  }

  Matrix<T> pinv(T eps=-1){
    if(eps<0){
    eps=1.0;
    while(eps+(T)1.0>1.0) eps*=0.5;
    eps=sqrtf(eps);
  }
    Matrix<T> M_r(dim[1],dim[0]);
    tpinv(data_ptr,dim[0],dim[1],eps,M_r.data_ptr);
    this->Resize(0,0);
    return M_r;
  }

};

template <class T>
class Permutation{

public:
  std::vector<size_t> perm;
  std::vector<T> scal;

  Permutation(){}

  Permutation(size_t size){
    perm.resize(size);
    scal.resize(size);
    std::iota(perm.begin(), perm.end(), 0.);
    std::fill(scal.begin(), scal.end(), 1.);
  }

  static Permutation<T> RandPerm(size_t size){
    Permutation<T> P(size);
    for(size_t i=0;i<size;i++){
      P.perm[i]=rand()%size;
      for(size_t j=0;j<i;j++)
	if(P.perm[i]==P.perm[j]){ i--; break; }
      P.scal[i]=((T)rand())/RAND_MAX;
    }
    return P;
  }

  Matrix<T> GetMatrix() const{
    size_t size=perm.size();
    Matrix<T> M_r(size,size,NULL);
    for(size_t i=0;i<size;i++)
      for(size_t j=0;j<size;j++)
	M_r[i][j]=(perm[j]==i?scal[j]:0.0);
    return M_r;
  }

  size_t Dim() const{
    return perm.size();
  }

  Permutation<T> Transpose(){
    size_t size=perm.size();
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0;i<size;i++){
      perm_r[perm[i]]=i;
      scal_r[perm[i]]=scal[i];
    }
    return P_r;
  }

  Permutation<T> operator*(const Permutation<T>& P){
    size_t size=perm.size();
    assert(P.Dim()==size);
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0;i<size;i++){
      perm_r[i]=perm[P.perm[i]];
      scal_r[i]=scal[P.perm[i]]*P.scal[i];
    }
    return P_r;
  }

  Matrix<T> operator*(const Matrix<T>& M){
    if(Dim()==0) return M;
    assert(M.Dim(0)==Dim());
    size_t d0=M.Dim(0);
    size_t d1=M.Dim(1);
    Matrix<T> M_r(d0,d1,NULL);
    for(size_t i=0;i<d0;i++){
      const T s=scal[i];
      const T* M_=M[i];
      T* M_r_=M_r[perm[i]];
      for(size_t j=0;j<d1;j++)
	M_r_[j]=M_[j]*s;
    }
    return M_r;
  }

  template <class Y>
  friend Matrix<Y> operator*(const Matrix<Y>& M, const Permutation<Y>& P);

};

template <class T>
Matrix<T> operator*(const Matrix<T>& M, const Permutation<T>& P){
  if(P.Dim()==0) return M;
  assert(M.Dim(1)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  Matrix<T> M_r(d0,d1,NULL);
  for(size_t i=0;i<d0;i++){
    const size_t* perm_=&(P.perm[0]);
    const T* scal_=&(P.scal[0]);
    const T* M_=M[i];
    T* M_r_=M_r[i];
    for(size_t j=0;j<d1;j++)
      M_r_[j]=M_[perm_[j]]*scal_[j];
  }
  return M_r;
}

}//end namespace

#endif //_PVFMM_MATRIX_HPP_
