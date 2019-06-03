#ifndef helmholtz_h
#define helmholtz_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "precompute_helmholtz.h"

extern "C" {
  void cgemv_(char* trans, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* lda, std::complex<float>* x,
              int* incx, std::complex<float>* beta, std::complex<float>* y, int* incy);
  void zgemv_(char* trans, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* lda, std::complex<double>* x,
              int* incx, std::complex<double>* beta, std::complex<double>* y, int* incy);
}

namespace exafmm_t {
  void gemv(int m, int n, complex_t* A, complex_t* x, complex_t* y);

  void potential_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void gradient_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

  void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out);

  void P2M(std::vector<Node*>& leafs);

  void M2M(Node* node);

  void L2L(Node* node);

  void L2P(std::vector<Node*>& leafs);

  void P2L(Nodes& nodes);

  void M2P(std::vector<Node*>& leafs);

  void P2P(std::vector<Node*>& leafs);

  void M2L_setup(std::vector<Node*>& nonleafs);

  void hadamard_product(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out, std::vector<AlignedVec>& matrix_M2L);

  void fft_up_equiv(std::vector<size_t>& fft_vec, ComplexVec& all_up_equiv, AlignedVec& fft_in);

  void ifft_dn_check(std::vector<size_t>& ifft_vec, AlignedVec& fft_out, ComplexVec& all_dn_equiv);

  void M2L(Nodes& nodes);
}//end namespace
#endif
