#ifndef laplace_h
#define laplace_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "precompute_laplace.h"

extern "C" {
  void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x,
              int* incx, float* beta, float* y, int* incy);
  void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x,
              int* incx, double* beta, double* y, int* incy);
}

namespace exafmm_t {
  void gemv(int m, int n, real_t* A, real_t* x, real_t* y);

  void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

  void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

  void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);

  void P2M(NodePtrs& leafs);

  void M2M(Node* node);

  void L2L(Node* node);

  void L2P(NodePtrs& leafs);

  void P2L(Nodes& nodes);

  void M2P(NodePtrs& leafs);

  void P2P(NodePtrs& leafs);

  void M2L_setup(NodePtrs& nonleafs);

  void hadamard_product(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out);

  void fft_up_equiv(std::vector<size_t>& fft_vec,
                   RealVec& all_up_equiv, AlignedVec& fft_in);

  void ifft_dn_check(std::vector<size_t>& ifft_vec,
                       AlignedVec& fft_out, RealVec& all_dn_equiv);

  void M2L(Nodes& nodes);

}//end namespace
#endif
