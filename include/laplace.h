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
  //! A derived FMM class for Laplace kernel.
  class LaplaceFMM : public FMM {
    using Body_t = Body<real_t>;
    using Bodies_t = Bodies<real_t>;
    using Node_t = Node<real_t>;
    using Nodes_t = Nodes<real_t>;
    using NodePtrs_t = NodePtrs<real_t>;

  public:
    RealVec matrix_UC2E_U;  //!< First component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    RealVec matrix_UC2E_V;  //!< Second component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    RealVec matrix_DC2E_U;  //!< First component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    RealVec matrix_DC2E_V;  //!< Second component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    std::vector<RealVec> matrix_M2M;     //!< The pseudo-inverse of M2M kernel matrix.
    std::vector<RealVec> matrix_L2L;     //!< The pseudo-inverse of L2L kernel matrix.
    std::vector<AlignedVec> matrix_M2L;  //!< The pseudo-inverse of M2L kernel matrix.
    M2LData m2ldata;

    LaplaceFMM() {}
    LaplaceFMM(int p_, int ncrit_, int depth_) : FMM(p_, ncrit_, depth_) {}

    void gemv(int m, int n, real_t* A, real_t* x, real_t* y);
    void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C);
    void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);
    RealVec transpose(RealVec& vec, int m, int n);

    /**
     * @brief Compute potentials at targets induced by sources directly.
     * 
     * @param src_coord Vector of coordinates of sources.
     * @param src_value Vector of charges of sources.
     * @param trg_coord Vector of coordinates of targets.
     * @param trg_value Vector of potentials of targets.
     */
    void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    /**
     * @brief Compute potentials and gradients at targets induced by sources directly.
     * 
     * @param src_coord Vector of coordinates of sources.
     * @param src_value Vector of charges of sources.
     * @param trg_coord Vector of coordinates of targets.
     * @param trg_value Vector of potentials of targets.
     */
    void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value);

    /**
     * @brief Create a kernel matrix.
     * 
     * @param r_src Pointer to coordinates of sources.
     * @param src_cnt Number of sources.
     * @param r_trg Pointer to coordinates of targets.
     * @param trg_cnt Number of targets.
     * @param k_out Pointer to kernel matrix.
     */
    void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);
  
    //! Initialize the precomputation matrices.
    void initialize_matrix();

    //! Pre-compute matrix_UC2E_U, matrix_UC2E_V, matrix_DC2E_U and matrix_DC2E_U.
    void precompute_check2equiv();

    //! Pre-compute matrix_M2M and matrix_L2L.
    void precompute_M2M();

    //! Pre-compute matrix_M2L.
    void precompute_M2L(std::vector<std::vector<int>>& parent2child);

    //! Load precomputation matrices from saved file to memory.
    bool load_matrix();

    //! Save precomputation matrices to file.
    void save_matrix();

    //! Pre-compute all matrices.
    void precompute();

    //! P2M operator.
    void P2M(NodePtrs_t& leafs);

    //! M2M operator.
    void M2M(Node_t* node);

    //! L2L operator.
    void L2L(Node_t* node);

    //! L2P operator.
    void L2P(NodePtrs_t& leafs);

    //! P2L operator.
    void P2L(Nodes_t& nodes);

    //! M2P operator.
    void M2P(NodePtrs_t& leafs);

    //! P2P operator.
    void P2P(NodePtrs_t& leafs);

    void M2L_setup(NodePtrs_t& nonleafs);

    void hadamard_product(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                          AlignedVec& fft_in, AlignedVec& fft_out);

    void fft_up_equiv(std::vector<size_t>& fft_offset, RealVec& all_up_equiv, AlignedVec& fft_in);

    void ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal, AlignedVec& fft_out, RealVec& all_dn_equiv);
    
    //! M2L operator.
    void M2L(Nodes_t& nodes);
    
    /**
     * @brief Evaluate upward equivalent charges for all nodes in a post-order traversal.
     * 
     * @param nodes Vector of all nodes.
     * @param leafs Vector of pointers to leaf nodes.
     */
    void upward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    /**
     * @brief Evaluate potentials and gradients for all targets in a pre-order traversal.
     * 
     * @param nodes Vector of all nodes.
     * @param leafs Vector of pointers to leaf nodes.
     */
    void downward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    /**
     * @brief Calculate the relative error of potentials and gradients in L2-norm.
     * 
     * @param leafs Vector of pointers to leaf nodes.
     * @return RealVec A two-element vector: potential error and gradient error.
     */
    RealVec verify(NodePtrs_t& leafs);
  };
}  // end namespace exafmm_t
#endif
