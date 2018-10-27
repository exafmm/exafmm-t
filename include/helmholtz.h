#ifndef helmholtz_h
#define helmholtz_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"

extern "C" {
  void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA, float* A,
              int* LDA, float* B, int* LDB, float* BETA, float* C, int* LDC);
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
              int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  void scgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<float>* ALPHA, float* A,
              int* LDA, std::complex<float>* B, int* LDB, std::complex<float>* BETA, std::complex<float>* C, int* LDC);
  void dzgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<double>* ALPHA, double* A,
              int* LDA, std::complex<double>* B, int* LDB, std::complex<double>* BETA, std::complex<double>* C, int* LDC);
  void cgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<float>* ALPHA, std::complex<float>* A,
              int* LDA, std::complex<float>* B, int* LDB, std::complex<float>* BETA, std::complex<float>* C, int* LDC);
  void zgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<double>* ALPHA, std::complex<double>* A,
              int* LDA, std::complex<double>* B, int* LDB, std::complex<double>* BETA, std::complex<double>* C, int* LDC);
  void sgesvd_(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
               float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO);
  void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
               double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
  void cgesvd_(char *JOBU, char *JOBVT, int *M, int *N, std::complex<float> *A, int *LDA,
               float *S, std::complex<float> *U, int *LDU, std::complex<float> *VT, int *LDVT, std::complex<float> *WORK, int *LWORK, float *RWORK, int *INFO);
  void zgesvd_(char *JOBU, char *JOBVT, int *M, int *N, std::complex<double> *A, int *LDA,
               double *S, std::complex<double> *U, int *LDU, std::complex<double> *VT, int *LDVT, std::complex<double> *WORK, int *LWORK, double *RWORK, int *INFO);
}

namespace exafmm_t {
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);

  void gemm(int m, int n, int k, complex_t* A, real_t* B, complex_t* C);

  void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C);

  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);

  void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT);

  RealVec transpose(RealVec& vec, int m, int n);

  ComplexVec transpose(ComplexVec& vec, int m, int n);

  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out);

  void P2M(std::vector<Node*>& leafs);

  void M2M(Node* node);

  void L2L(Node* node);

  void L2P(std::vector<Node*>& leafs);

  void P2L(Nodes& nodes);

  void M2P(std::vector<Node*>& leafs);

  void P2P(std::vector<Node*>& leafs);

  void M2LSetup(std::vector<Node*>& nonleafs);

  void M2LListHadamard(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out);

  void FFT_UpEquiv(std::vector<size_t>& fft_vec, RealVec& fft_scal,
                   ComplexVec& input_data, AlignedVec& fft_in);

  void FFT_Check2Equiv(std::vector<size_t>& ifft_vec, RealVec& ifft_scal,
                       AlignedVec& fft_out, ComplexVec& output_data);

  void M2L(M2LData& M2Ldata, Nodes& nodes);
}//end namespace
#endif
