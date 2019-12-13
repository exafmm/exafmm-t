#ifndef helmholtz_h
#define helmholtz_h
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

extern "C" {
  void cgemv_(char* trans, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* lda, std::complex<float>* x,
              int* incx, std::complex<float>* beta, std::complex<float>* y, int* incy);
  void zgemv_(char* trans, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* lda, std::complex<double>* x,
              int* incx, std::complex<double>* beta, std::complex<double>* y, int* incy);
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
  //! A derived FMM class for Helmholtz kernel.
  class HelmholtzFMM : public FMM {
    using Body_t = Body<complex_t>;
    using Bodies_t = Bodies<complex_t>;
    using Node_t = Node<complex_t>;
    using Nodes_t = Nodes<complex_t>;
    using NodePtrs_t = NodePtrs<complex_t>;

  public:
    real_t wavek;   //!< Wave number k.
    std::vector<ComplexVec> matrix_UC2E_U;
    std::vector<ComplexVec> matrix_UC2E_V;
    std::vector<ComplexVec> matrix_DC2E_U;
    std::vector<ComplexVec> matrix_DC2E_V;
    std::vector<std::vector<ComplexVec>> matrix_M2M;
    std::vector<std::vector<ComplexVec>> matrix_L2L;
    std::vector<M2LData> m2ldata;

    void gemv(int m, int n, complex_t* A, complex_t* x, complex_t* y);
    void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C);
    void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT);
    ComplexVec transpose(ComplexVec& vec, int m, int n);
    ComplexVec conjugate_transpose(ComplexVec& vec, int m, int n);

    void potential_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

    void gradient_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value);

    void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out);

    void initialize_matrix();

    void precompute_check2equiv();

    void precompute_M2M();

    void precompute_M2L(std::ofstream& file, std::vector<std::vector<int>>& parent2child);

    bool load_matrix();

    void save_matrix(std::ofstream& file);

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
                         AlignedVec& fft_in, AlignedVec& fft_out, std::vector<AlignedVec>& matrix_M2L);

    void fft_up_equiv(std::vector<size_t>& fft_vec, ComplexVec& all_up_equiv, AlignedVec& fft_in);

    void ifft_dn_check(std::vector<size_t>& ifft_vec, AlignedVec& fft_out, ComplexVec& all_dn_equiv);

    void M2L(Nodes_t& nodes);

    void upward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    void downward_pass(Nodes_t& nodes, NodePtrs_t& leafs);

    RealVec verify(NodePtrs_t& leafs);
  };
}  // end namespace exafmm_t
#endif
