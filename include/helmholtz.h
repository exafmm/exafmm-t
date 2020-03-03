#ifndef helmholtz_h
#define helmholtz_h
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"
#include "timer.h"

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

    HelmholtzFMM() {}
    HelmholtzFMM(int p_, int ncrit_, int depth_, real_t wavek_) : FMM(p_, ncrit_, depth_) { wavek = wavek_;}

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
