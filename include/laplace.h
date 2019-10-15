#ifndef laplace_h
#define laplace_h
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

extern "C" {
  void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x,
              int* incx, float* beta, float* y, int* incy);
  void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x,
              int* incx, double* beta, double* y, int* incy);
  void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a,
              int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
  void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a,
              int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
  void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda,
               float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info);
  void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
               double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info);
}
namespace exafmm_t {
  class LaplaceFMM : public FMM {
    using Body_t = Body<real_t>;
    using Bodies_t = Bodies<real_t>;
    using Node_t = Node<real_t>;
    using Nodes_t = Nodes<real_t>;
    using NodePtrs_t = NodePtrs<real_t>;

  public:
    RealVec matrix_UC2E_U;
    RealVec matrix_UC2E_V;
    RealVec matrix_DC2E_U;
    RealVec matrix_DC2E_V;
    std::vector<RealVec> matrix_M2M;
    std::vector<RealVec> matrix_L2L;
    std::vector<AlignedVec> matrix_M2L;

    void gemv(int m, int n, real_t* A, real_t* x, real_t* y);
    void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);
    void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);
    RealVec transpose(RealVec& vec, int m, int n);

    void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);
  
    void initialize_matrix();

    void precompute_check2equiv();

    void precompute_M2M();

    void precompute_M2L(std::vector<std::vector<int>>& parent2child);

    bool load_matrix();

    void save_matrix();

    void precompute();

    void P2M(NodePtrs_t& leafs);

    void M2M(Node_t* node);

    void L2L(Node_t* node);

    void L2P(NodePtrs_t& leafs);

    void P2L(Nodes_t& nodes);

    void M2P(NodePtrs_t& leafs);

    void P2P(NodePtrs_t& leafs);

    void M2L_setup(NodePtrs_t& nonleafs);

    void hadamard_product(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                          AlignedVec& fft_in, AlignedVec& fft_out);

    void fft_up_equiv(std::vector<size_t>& fft_offset, RealVec& all_up_equiv, AlignedVec& fft_in);

    void ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal, AlignedVec& fft_out, RealVec& all_dn_equiv);

    void M2L(Nodes_t& nodes);

    void upward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    void downward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    RealVec verify(NodePtrs_t& leafs);
  };
}  // end namespace exafmm_t
#endif
