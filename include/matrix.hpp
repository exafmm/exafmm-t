#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_
#define MEM_ALIGN 64
#include "profile.hpp"

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

namespace pvfmm {
template <class T>
class Permutation;

template <class T>
class Matrix {
 public:
  T* data_ptr;
  int dim[2];

  Matrix() {
    dim[0]=0;
    dim[1]=0;
    data_ptr=NULL;
  }

  Matrix(int dim1, int dim2, T* data_=NULL) {
    dim[0]=dim1;
    dim[1]=dim2;
    if(dim[0]*dim[1]>0) {
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim[0]*dim[1]*sizeof(T));
      if(data_!=NULL) memcpy(data_ptr, data_, dim[0]*dim[1]*sizeof(T));
    } else data_ptr=NULL;
  }

  Matrix(const Matrix<T>& M) {
    dim[0]=M.dim[0];
    dim[1]=M.dim[1];
    if(dim[0]*dim[1]>0) {
      int err = posix_memalign((void**)&data_ptr, MEM_ALIGN, dim[0]*dim[1]*sizeof(T));
      memcpy(data_ptr, M.data_ptr, dim[0]*dim[1]*sizeof(T));
    } else
      data_ptr=NULL;
  }

  ~Matrix() {
    if(data_ptr!=NULL) free(data_ptr);
    data_ptr=NULL;
    dim[0]=0;
    dim[1]=0;
  }

  void Swap(Matrix<T>& M) {
    int dim_[2]= {dim[0], dim[1]};
    T* data_ptr_=data_ptr;
    dim[0]=M.dim[0];
    dim[1]=M.dim[1];
    data_ptr=M.data_ptr;
    M.dim[0]=dim_[0];
    M.dim[1]=dim_[1];
    M.data_ptr=data_ptr_;
  }

  void ReInit(int dim1, int dim2, T* data_=NULL) {
    if(dim[0]*dim[1]>=dim1*dim2) {
      dim[0]=dim1;
      dim[1]=dim2;
      if(data_) memcpy(data_ptr, data_, dim[0]*dim[1]*sizeof(T));
    } else {
      Matrix<T> tmp(dim1, dim2, data_);
      this->Swap(tmp);
    }
  }

  int Dim(size_t i) const {
    return dim[i];
  }

  void Resize(int i, int j) {
    if(dim[0]*dim[1]>=i*j) {
      dim[0]=i;
      dim[1]=j;
    } else ReInit(i, j);
  }

  void SetZero() {
    if(dim[0]*dim[1])
      memset(data_ptr, 0, dim[0]*dim[1]*sizeof(T));
  }

  Matrix<T>& operator=(const Matrix<T>& M) {
    if(this!=&M) {
      if(dim[0]*dim[1]<M.dim[0]*M.dim[1])
        ReInit(M.dim[0], M.dim[1]);
      dim[0]=M.dim[0];
      dim[1]=M.dim[1];
      memcpy(data_ptr, M.data_ptr, dim[0]*dim[1]*sizeof(T));
    }
    return *this;
  }

  inline T* operator[](int i) const {
    assert(i<dim[0]);
    return &data_ptr[i*dim[1]];
  }

  friend Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
    assert(A.dim[1] == B.dim[0]);
    char transA = 'N', transB = 'N';
    T alpha = 1.0, beta = 0.0;
    Matrix<T> M_r(A.dim[0], B.dim[1]);
#if FLOAT
    sgemm_(&transA, &transB, (int*)&B.dim[1], (int*)&A.dim[0], (int*)&A.dim[1], &alpha, B.data_ptr,
           (int*)&B.dim[1], A.data_ptr, (int*)&A.dim[1], &beta, M_r.data_ptr, (int*)&M_r.dim[1]);
#else
    dgemm_(&transA, &transB, (int*)&B.dim[1], (int*)&A.dim[0], (int*)&A.dim[1], &alpha, B.data_ptr,
           (int*)&B.dim[1], A.data_ptr, (int*)&A.dim[1], &beta, M_r.data_ptr, (int*)&M_r.dim[1]);
#endif
    return M_r;
  }

  static void GEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0) {
    if(A.Dim(0)*A.Dim(1)==0 || B.Dim(0)*B.Dim(1)==0) return;
    assert(M_r.dim[0] == A.dim[0]);
    assert(M_r.dim[1] == B.dim[1]);
    M_r = A * B;
  }

#define B1 128
#define B2 32

  Matrix<T> Transpose() {
    Matrix<T>& M=*this;
    int d0=M.dim[0];
    int d1=M.dim[1];
    Matrix<T> M_r(d1, d0, NULL);
    const int blk0=((d0+B1-1)/B1);
    const int blk1=((d1+B1-1)/B1);
    const int blks=blk0*blk1;
    for(int k=0; k<blks; k++) {
      int i=(k%blk0)*B1;
      int j=(k/blk0)*B1;
      int d0_=i+B1;
      if(d0_>=d0) d0_=d0;
      int d1_=j+B1;
      if(d1_>=d1) d1_=d1;
      for(int ii=i; ii<d0_; ii+=B2)
        for(int jj=j; jj<d1_; jj+=B2) {
          int d0__=ii+B2;
          if(d0__>=d0) d0__=d0;
          int d1__=jj+B2;
          if(d1__>=d1) d1__=d1;
          for(int iii=ii; iii<d0__; iii++)
            for(int jjj=jj; jjj<d1__; jjj++)
              M_r[jjj][iii]=M[iii][jjj];
        }
    }
    return M_r;
  }

#undef B2
#undef B1

  void SVD(Matrix<T>& tU, Matrix<T>& tS, Matrix<T>& tVT) {
    pvfmm::Matrix<T>& M=*this;
    pvfmm::Matrix<T> M_=M;
    int n=M.Dim(0);
    int m=M.Dim(1);
    int k = (m<n?m:n);
    tU.Resize(n, k);
    tU.SetZero();
    tS.Resize(k, k);
    tS.SetZero();
    tVT.Resize(k, m);
    tVT.SetZero();
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);
    T* wsbuf;
    int err = posix_memalign((void**)&wsbuf, MEM_ALIGN, wssize*sizeof(T));
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf,
            &wssize, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf,
            &wssize, &INFO);
#endif
    //svd(&JOBU, &JOBVT, &m, &n, &M[0][0], &m, &tS[0][0], &tVT[0][0], &m, &tU[0][0], &k, wsbuf, &wssize, &INFO);
    free(wsbuf);
    if(INFO!=0) std::cout<<INFO<<'\n';
    assert(INFO==0);
    for(int i=1; i<k; i++) {
      tS[i][i]=tS[0][i];
      tS[0][i]=0;
    }
  }
};

template <class T>
class Permutation {

 public:
  std::vector<size_t> perm;
  std::vector<T> scal;

  Permutation() {}

  Permutation(size_t size) {
    perm.resize(size);
    scal.resize(size);
    std::iota(perm.begin(), perm.end(), 0.);
    std::fill(scal.begin(), scal.end(), 1.);
  }

  static Permutation<T> RandPerm(size_t size) {
    Permutation<T> P(size);
    for(size_t i=0; i<size; i++) {
      P.perm[i]=rand()%size;
      for(size_t j=0; j<i; j++)
        if(P.perm[i]==P.perm[j]) {
          i--;
          break;
        }
      P.scal[i]=((T)rand())/RAND_MAX;
    }
    return P;
  }

  Matrix<T> GetMatrix() const {
    size_t size=perm.size();
    Matrix<T> M_r(size, size, NULL);
    for(size_t i=0; i<size; i++)
      for(size_t j=0; j<size; j++)
        M_r[i][j]=(perm[j]==i?scal[j]:0.0);
    return M_r;
  }

  size_t Dim() const {
    return perm.size();
  }

  Permutation<T> Transpose() {
    size_t size=perm.size();
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0; i<size; i++) {
      perm_r[perm[i]]=i;
      scal_r[perm[i]]=scal[i];
    }
    return P_r;
  }

  Permutation<T> operator*(const Permutation<T>& P) {
    size_t size=perm.size();
    assert(P.Dim()==size);
    Permutation<T> P_r(size);
    std::vector<size_t>& perm_r=P_r.perm;
    std::vector<T>& scal_r=P_r.scal;
    for(size_t i=0; i<size; i++) {
      perm_r[i]=perm[P.perm[i]];
      scal_r[i]=scal[P.perm[i]]*P.scal[i];
    }
    return P_r;
  }

  Matrix<T> operator*(const Matrix<T>& M) {
    if(Dim()==0) return M;
    assert(M.Dim(0)==Dim());
    size_t d0=M.Dim(0);
    size_t d1=M.Dim(1);
    Matrix<T> M_r(d0, d1, NULL);
    for(size_t i=0; i<d0; i++) {
      const T s=scal[i];
      const T* M_=M[i];
      T* M_r_=M_r[perm[i]];
      for(size_t j=0; j<d1; j++)
        M_r_[j]=M_[j]*s;
    }
    return M_r;
  }

  template <class Y>
  friend Matrix<Y> operator*(const Matrix<Y>& M, const Permutation<Y>& P);
};

template <class T>
Matrix<T> operator*(const Matrix<T>& M, const Permutation<T>& P) {
  if(P.Dim()==0) return M;
  assert(M.Dim(1)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);
  Matrix<T> M_r(d0, d1, NULL);
  for(size_t i=0; i<d0; i++) {
    const size_t* perm_=&(P.perm[0]);
    const T* scal_=&(P.scal[0]);
    const T* M_=M[i];
    T* M_r_=M_r[i];
    for(size_t j=0; j<d1; j++)
      M_r_[j]=M_[perm_[j]]*scal_[j];
  }
  return M_r;
}
}//end namespace
#endif //_PVFMM_MATRIX_HPP_
