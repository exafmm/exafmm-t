#ifndef laplace_h
#define laplace_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

extern "C" {
  void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x,
              int* incx, float* beta, float* y, int* incy);
  void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x,
              int* incx, double* beta, double* y, int* incy);
}

namespace exafmm_t {
namespace laplace {

  extern RealVec matrix_UC2E_U, matrix_UC2E_V;
  extern RealVec matrix_DC2E_U, matrix_DC2E_V;
  extern std::vector<RealVec> matrix_M2M, matrix_L2L;
  extern std::vector<AlignedVec> matrix_M2L;

  using Body = Body<real_t>;
  using Bodies = Bodies<real_t>;
  using Node = Node<real_t>;
  using Nodes = Nodes<real_t>;
  using NodePtrs = NodePtrs<real_t>;

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

  void upward_pass(Nodes& nodes, NodePtrs& leafs);

  void downward_pass(Nodes& nodes, NodePtrs& leafs);

  RealVec verify(NodePtrs& leafs);
}  // end namespace laplace
}  // end namespace exafmm_t
#endif
