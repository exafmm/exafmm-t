#ifndef laplace_c_h
#define laplace_c_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"

extern "C" {
  void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a,
              int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a,
              int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
  void scgemv_(char* trans, int* m, int* n, std::complex<float>* alpha, float* a, int* lda, std::complex<float>* x,
              int* incx, std::complex<float>* beta, std::complex<float>* y, int* incy);
  void dzgemv_(char* trans, int* m, int* n, std::complex<double>* alpha, double* a, int* lda, std::complex<double>* x,
              int* incx, std::complex<double>* beta, std::complex<double>* y, int* incy);
  void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda,
               float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
  void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
               double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
}

namespace exafmm_t {
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);

  void gemv(int m, int n, real_t* A, complex_t* x, complex_t* y);

  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);

  RealVec transpose(RealVec& vec, int m, int n);

  void potentialP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);

  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out);

  void P2M(NodePtrs& leafs);

  void M2M(Node* node);

  void L2L(Node* node);

  void L2P(NodePtrs& leafs);

  void P2L(Nodes& nodes);

  void M2P(NodePtrs& leafs);

  void P2P(NodePtrs& leafs);

  void M2LSetup(NodePtrs& nonleafs);

  void M2LListHadamard(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out);

  void FFT_UpEquiv(std::vector<size_t>& fft_vec, RealVec& fft_scal,
                   ComplexVec& input_data, AlignedVec& fft_in);

  void FFT_Check2Equiv(std::vector<size_t>& ifft_vec, RealVec& ifft_scal,
                       AlignedVec& fft_out, ComplexVec& output_data);

  void M2L(M2LData& M2Ldata, Nodes& nodes);
}//end namespace
#endif
