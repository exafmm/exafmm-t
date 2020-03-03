#ifndef math_wrapper_h
#define math_wrapper_h
#include <complex>
#include "exafmm_t.h"

using std::complex;

extern "C" {
  void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda,
              float* x, int* incx, float* beta, float* y, int* incy);

  void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda,
              double* x, int* incx, double* beta, double* y, int* incy);

  void cgemv_(char* trans, int* m, int* n, complex<float>* alpha, complex<float>* a, int* lda,
              complex<float>* x, int* incx, complex<float>* beta, complex<float>* y, int* incy);

  void zgemv_(char* trans, int* m, int* n, complex<double>* alpha, complex<double>* a, int* lda,
              complex<double>* x, int* incx, complex<double>* beta, complex<double>* y, int* incy);

  void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a,
              int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);

  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a,
              int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);

  void cgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, complex<float>* ALPHA, complex<float>* A,
              int* LDA, complex<float>* B, int* LDB, complex<float>* BETA, complex<float>* C, int* LDC);

  void zgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, complex<double>* ALPHA, complex<double>* A,
              int* LDA, complex<double>* B, int* LDB, complex<double>* BETA, complex<double>* C, int* LDC);

  void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s, float *u,
               int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);

  void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u,
               int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);

  void cgesvd_(char *JOBU, char *JOBVT, int *M, int *N, complex<float> *A, int *LDA,
               float *S, complex<float> *U, int *LDU, complex<float> *VT, int *LDVT,
               complex<float> *WORK, int *LWORK, float *RWORK, int *INFO);

  void zgesvd_(char *JOBU, char *JOBVT, int *M, int *N, complex<double> *A, int *LDA,
               double *S, complex<double> *U, int *LDU, complex<double> *VT, int *LDVT,
               complex<double> *WORK, int *LWORK, double *RWORK, int *INFO);
}

namespace exafmm_t {
  //! blas gemv with row major data
  void gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
    char trans = 'T';
    real_t alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
#if FLOAT
    sgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    dgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }

  // complex gemv by blas lib
  void gemv(int m, int n, complex_t* A, complex_t* x, complex_t* y) {
    char trans = 'T';
    complex_t alpha(1., 0.), beta(0.,0.);
    int incx = 1, incy = 1;
#if FLOAT
    cgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    zgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }
  
  //! blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  // complex gemm by blas lib
  void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C) {
    char transA = 'N', transB = 'N';
    complex_t alpha(1., 0.), beta(0.,0.);
#if FLOAT
    cgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    zgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    RealVec WORK(LWORK);
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    ComplexVec WORK(LWORK);
    RealVec RWORK(5*k);
#if FLOAT
    cgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
#else
    zgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }
  
  RealVec transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  ComplexVec transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  ComplexVec conjugate_transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = std::conj(vec[i*n+j]);
      }
    }
    return temp;
  }

}  // end namespace exafmm_t
#endif
