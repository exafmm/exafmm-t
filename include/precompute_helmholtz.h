#ifndef precompute_helmholtz_h
#define precompute_helmholtz_h
#include <fstream>
#include <string>
#include "exafmm_t.h"
#include "geometry.h"
#include "helmholtz.h"

extern "C" {
  void cgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<float>* ALPHA, std::complex<float>* A,
              int* LDA, std::complex<float>* B, int* LDB, std::complex<float>* BETA, std::complex<float>* C, int* LDC);
  void zgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<double>* ALPHA, std::complex<double>* A,
              int* LDA, std::complex<double>* B, int* LDB, std::complex<double>* BETA, std::complex<double>* C, int* LDC);
  void cgesvd_(char *JOBU, char *JOBVT, int *M, int *N, std::complex<float> *A, int *LDA,
               float *S, std::complex<float> *U, int *LDU, std::complex<float> *VT, int *LDVT, std::complex<float> *WORK, int *LWORK, float *RWORK, int *INFO);
  void zgesvd_(char *JOBU, char *JOBVT, int *M, int *N, std::complex<double> *A, int *LDA,
               double *S, std::complex<double> *U, int *LDU, std::complex<double> *VT, int *LDVT, std::complex<double> *WORK, int *LWORK, double *RWORK, int *INFO);
}

namespace exafmm_t {
  void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C);

  void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT);

  ComplexVec transpose(ComplexVec& vec, int m, int n);

  ComplexVec conjugate_transpose(ComplexVec& vec, int m, int n);

  void initialize_matrix();

  void precompute_check2equiv();

  void precompute_M2M();

  void precompute_M2Lhelper();

  void precompute_M2L();

  bool load_matrix();

  void save_matrix();

  void precompute();
}//end namespace
#endif
