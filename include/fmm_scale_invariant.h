#ifndef fmm_scale_invariant_h
#define fmm_scale_invariant_h
#include <cstring>      // std::memset
#include <fstream>      // std::ofstream
#include <type_traits>  // std::is_same
#include "fmm_base.h"
#include "intrinsics.h"
#include "math_wrapper.h"

namespace exafmm_t {
  template <typename T>
  class FmmScaleInvariant : public FmmBase<T> {
    /** For the variables from base class that do not template parameter T,
     *  we need to use this-> to tell compilers to lookup nondependent names
     *  in the base class. Eg. p, nsurf, r0, kernel_matrix etc.
     *  https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members/
     */

  public:
    /* precomputation matrices */
    std::vector<T> matrix_UC2E_U;  //!< First component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    std::vector<T> matrix_UC2E_V;  //!< Second component of the pseudo-inverse of upward check to upward equivalent kernel matrix.
    std::vector<T> matrix_DC2E_U;  //!< First component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    std::vector<T> matrix_DC2E_V;  //!< Second component of the pseudo-inverse of downward check to downward equivalent kernel matrix.
    std::vector<std::vector<T>> matrix_M2M;     //!< The pseudo-inverse of M2M kernel matrix.
    std::vector<std::vector<T>> matrix_L2L;     //!< The pseudo-inverse of L2L kernel matrix.
    std::vector<AlignedVec> matrix_M2L;  //!< The pseudo-inverse of M2L kernel matrix.

    M2LData m2ldata;

    /* constructors */
    FmmScaleInvariant() {}
    FmmScaleInvariant(int p_, int ncrit_, int depth_, std::string filename_=std::string()) : FmmBase<T>(p_, ncrit_, depth_, filename_) {}

    /* precomputation */
    //! Setup the sizes of precomputation matrices
    void initialize_matrix() {
      size_t size = this->nfreq * 2 * NCHILD * NCHILD;  // size of each M2L precomputation matrix
      int& nsurf_ = this->nsurf;
      matrix_UC2E_U.resize(nsurf_*nsurf_);
      matrix_UC2E_V.resize(nsurf_*nsurf_);
      matrix_DC2E_U.resize(nsurf_*nsurf_);
      matrix_DC2E_V.resize(nsurf_*nsurf_);
      matrix_M2M.resize(REL_COORD[M2M_Type].size(), std::vector<T>(nsurf_*nsurf_));
      matrix_L2L.resize(REL_COORD[L2L_Type].size(), std::vector<T>(nsurf_*nsurf_));
      matrix_M2L.resize(REL_COORD[M2L_Type].size(), AlignedVec(size));    
    }

    //! Precompute M2M and L2L
    void precompute_M2M() {
      int& nsurf_ = this->nsurf;
      int npos = REL_COORD[M2M_Type].size();  // number of relative positions
      int level = 0;
      real_t parent_coord[3] = {0, 0, 0};
      RealVec parent_up_check_surf = surface(this->p, this->r0, level, parent_coord, 2.95);
      real_t s = this->r0 * powf(0.5, level+1);
#pragma omp parallel for
      for (int i=0; i<npos; i++) {
        // compute kernel matrix
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(this->p, this->r0, level+1, child_coord, 1.05);
        std::vector<T> matrix_pc2ce(nsurf_*nsurf_);
        this->kernel_matrix(parent_up_check_surf, child_up_equiv_surf, matrix_pc2ce);
        // M2M
        std::vector<T> buffer(nsurf_*nsurf_);
        gemm(nsurf_, nsurf_, nsurf_, &matrix_UC2E_U[0], &matrix_pc2ce[0], &buffer[0]);
        gemm(nsurf_, nsurf_, nsurf_, &matrix_UC2E_V[0], &buffer[0], &(matrix_M2M[i][0]));
        // L2L
        matrix_pc2ce = transpose(matrix_pc2ce, nsurf_, nsurf_);
        gemm(nsurf_, nsurf_, nsurf_, &matrix_pc2ce[0], &matrix_DC2E_V[0], &buffer[0]);
        gemm(nsurf_, nsurf_, nsurf_, &buffer[0], &matrix_DC2E_U[0], &(matrix_L2L[i][0]));
      }
    }

    //! Precompute UC2UE and DC2DE matrices
    void precompute_check2equiv() {}

    //! Precompute M2L
    void precompute_M2L() {}

    //! Save precomputation matrices
    void save_matrix() {
      std::remove(this->filename.c_str());
      std::ofstream file(this->filename, std::ofstream::binary);
      // r0
      file.write(reinterpret_cast<char*>(&this->r0), sizeof(real_t));
      size_t size = this->nsurf * this->nsurf;
      // UC2E, DC2E
      file.write(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(T));
      file.write(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(T));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(T));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(T));
      // M2M, L2L
      for (auto & vec : matrix_M2M) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
      }
      for (auto & vec : matrix_L2L) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
      }
      // M2L
      size = this->nfreq * 2 * NCHILD * NCHILD;
      for (auto & vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
      file.close(); 
    }

    //! Check and load precomputation matrices
    void load_matrix() {
      size_t size_M2L = this->nfreq * 2 * NCHILD * NCHILD;
      size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * this->nsurf * this->nsurf * sizeof(T) 
                       + REL_COORD[M2L_Type].size() * size_M2L * sizeof(real_t)
                       + 1 * sizeof(real_t);   // +1 denotes r0
      std::ifstream file(this->filename, std::ifstream::binary);
      if (file.good()) {
        file.seekg(0, file.end);
        if (size_t(file.tellg()) == file_size) {   // if file size is correct
          file.seekg(0, file.beg);  // move the position back to the beginning
          real_t r0_;
          file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
          if (this->r0 == r0_) {    // if radius match
            size_t size = this->nsurf * this->nsurf;
            // UC2E, DC2E
            file.read(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(T));
            file.read(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(T));
            file.read(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(T));
            file.read(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(T));
            // M2M, L2L
            for (auto & vec : matrix_M2M) {
              file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
            }
            for (auto & vec : matrix_L2L) {
              file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
            }
            // M2L
            for (auto & vec : matrix_M2L) {
              file.read(reinterpret_cast<char*>(&vec[0]), size_M2L*sizeof(real_t));
            }
            this->is_precomputed = true;
          }
        }
      }
      file.close();
    }

    //! Precompute
    void precompute() {
      initialize_matrix();
      load_matrix();
      if (!this->is_precomputed) {
        precompute_check2equiv();
        precompute_M2M();
        precompute_M2L();
        save_matrix();
      }
    }

    //! P2M operator
    void P2M(NodePtrs<T>& leafs) {
      int& nsurf_ = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> up_check_surf;
      up_check_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        up_check_surf[level].resize(nsurf_*3);
        up_check_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        real_t scale = pow(0.5, level);  // scaling factor of UC2UE precomputation matrix
        // calculate upward check potential induced by sources' charges
        RealVec check_coord(nsurf_*3);
        for (int k=0; k<nsurf_; k++) {
          check_coord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
          check_coord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
          check_coord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
        }
        this->potential_P2P(leaf->src_coord, leaf->src_value,
                            check_coord, leaf->up_equiv);
        // convert upward check potential to upward equivalent charge
        std::vector<T> buffer(nsurf_);
        std::vector<T> equiv(nsurf_);
        gemv(nsurf_, nsurf_, &matrix_UC2E_U[0], &(leaf->up_equiv[0]), &buffer[0]);
        gemv(nsurf_, nsurf_, &matrix_UC2E_V[0], &buffer[0], &equiv[0]);
        // scale the check-to-equivalent conversion (precomputation)
        for (int k=0; k<nsurf_; k++)
          leaf->up_equiv[k] = scale * equiv[k];
      }
    }

    //! L2P operator
    void L2P(NodePtrs<T>& leafs) {
      int& nsurf_ = this->nsurf;
      real_t c[3] = {0.0};
      std::vector<RealVec> dn_equiv_surf;
      dn_equiv_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        dn_equiv_surf[level].resize(nsurf_*3);
        dn_equiv_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        real_t scale = pow(0.5, level);
        // convert downward check potential to downward equivalent charge
        std::vector<T> buffer(nsurf_);
        std::vector<T> equiv(nsurf_);
        gemv(nsurf_, nsurf_, &matrix_DC2E_U[0], &(leaf->dn_equiv[0]), &buffer[0]);
        gemv(nsurf_, nsurf_, &matrix_DC2E_V[0], &buffer[0], &equiv[0]);
        // scale the check-to-equivalent conversion (precomputation)
        for (int k=0; k<nsurf_; k++)
          leaf->dn_equiv[k] = scale * equiv[k];
        // calculate targets' potential & gradient induced by downward equivalent charge
        RealVec equiv_coord(nsurf_*3);
        for (int k=0; k<nsurf_; k++) {
          equiv_coord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->x[0];
          equiv_coord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->x[1];
          equiv_coord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->x[2];
        }
        this->gradient_P2P(equiv_coord, leaf->dn_equiv,
                           leaf->trg_coord, leaf->trg_value);
      }
    }

    //! M2M operator
    void M2M(Node<T>* node) {
      int& nsurf_ = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      // evaluate parent's upward equivalent charge from child's upward equivalent charge
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf_);
          gemv(nsurf_, nsurf_, &(matrix_M2M[octant][0]), &child->up_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf_; k++) {
            node->up_equiv[k] += buffer[k];
          }
        }
      }
    }

    //! L2L operator
    void L2L(Node<T>* node) {
      int& nsurf_ = this->nsurf;
      if (node->is_leaf) return;
      // evaluate child's downward check potential from parent's downward check potential
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf_);
          gemv(nsurf_, nsurf_, &(matrix_L2L[octant][0]), &node->dn_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf_; k++)
            child->dn_equiv[k] += buffer[k];
        }
      }
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          L2L(node->children[octant]);
      }
#pragma omp taskwait
    }

    void M2L_setup(NodePtrs<T> nonleafs) {
      int& nsurf_ = this->nsurf;
      int npos = REL_COORD[M2L_Type].size();  // number of M2L relative positions

      // construct lists of source nodes and target nodes for M2L operator
      NodePtrs<T>& trg_nodes = nonleafs;
      std::set<Node<T>*> src_nodes_;
      for (size_t i=0; i<trg_nodes.size(); i++) {
        NodePtrs<T>& M2L_list = trg_nodes[i]->M2L_list;
        for (int k=0; k<npos; k++) {
          if (M2L_list[k]) {
            src_nodes_.insert(M2L_list[k]);
          }
        }
      }
      NodePtrs<T> src_nodes;
      auto it = src_nodes_.begin(); 
      for (; it!=src_nodes_.end(); it++) {
        src_nodes.push_back(*it);
      }

      // prepare the indices of src_nodes & trg_nodes in all_up_equiv & all_dn_equiv
      std::vector<size_t> fft_offset(src_nodes.size());
      std::vector<size_t> ifft_offset(trg_nodes.size());
      RealVec ifft_scale(trg_nodes.size());
      for (size_t i=0; i<src_nodes.size(); i++) {
        fft_offset[i] = src_nodes[i]->children[0]->idx * nsurf_;
      }
      for (size_t i=0; i<trg_nodes.size(); i++) {
        int level = trg_nodes[i]->level+1;
        ifft_offset[i] = trg_nodes[i]->children[0]->idx * nsurf_;
        ifft_scale[i] = powf(2.0, level);
      }

      // calculate interaction_offset_f & interaction_count_offset
      std::vector<size_t> interaction_offset_f;
      std::vector<size_t> interaction_count_offset;
      for (size_t i=0; i<src_nodes.size(); i++) {
        src_nodes[i]->idx_M2L = i;
      }
      size_t n_blk1 = trg_nodes.size() * sizeof(real_t) / CACHE_SIZE;
      if (n_blk1==0) n_blk1 = 1;
      size_t interaction_count_offset_ = 0;
      size_t fft_size = 2 * NCHILD * this->nfreq;
      for (size_t blk1=0; blk1<n_blk1; blk1++) {
        size_t blk1_start=(trg_nodes.size()* blk1   )/n_blk1;
        size_t blk1_end  =(trg_nodes.size()*(blk1+1))/n_blk1;
        for (int k=0; k<npos; k++) {
          for (size_t i=blk1_start; i<blk1_end; i++) {
            NodePtrs<T>& M2L_list = trg_nodes[i]->M2L_list;
            if (M2L_list[k]) {
              interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fft_size);   // node_in dspl
              interaction_offset_f.push_back(        i           * fft_size);   // node_out dspl
              interaction_count_offset_++;
            }
          }
          interaction_count_offset.push_back(interaction_count_offset_);
        }
      }
      m2ldata.fft_offset = fft_offset;
      m2ldata.ifft_offset = ifft_offset;
      m2ldata.ifft_scale = ifft_scale;
      m2ldata.interaction_offset_f = interaction_offset_f;
      m2ldata.interaction_count_offset = interaction_count_offset;
    }

    void hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                         AlignedVec& fft_in, AlignedVec& fft_out) {
      size_t fft_size = 2 * NCHILD * this->nfreq;
      AlignedVec zero_vec0(fft_size, 0.);
      AlignedVec zero_vec1(fft_size, 0.);

      size_t npos = matrix_M2L.size();
      size_t blk1_cnt = interaction_count_offset.size()/npos;
      int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
      std::vector<real_t*> IN_(BLOCK_SIZE*blk1_cnt*npos);
      std::vector<real_t*> OUT_(BLOCK_SIZE*blk1_cnt*npos);

      // initialize fft_out with zero
#pragma omp parallel for
      for (size_t i=0; i<fft_out.capacity()/fft_size; ++i) {
        std::memset(fft_out.data()+i*fft_size, 0, fft_size*sizeof(real_t));
      }

#pragma omp parallel for
      for (size_t interac_blk1=0; interac_blk1<blk1_cnt*npos; interac_blk1++) {
        size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
        size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
        size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
        for (size_t j=0; j<interac_cnt; j++) {
          IN_ [BLOCK_SIZE*interac_blk1 +j] = &fft_in[interaction_offset_f[(interaction_count_offset0+j)*2+0]];
          OUT_[BLOCK_SIZE*interac_blk1 +j] = &fft_out[interaction_offset_f[(interaction_count_offset0+j)*2+1]];
        }
        IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec0[0];
        OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec1[0];
      }

      for (size_t blk1=0; blk1<blk1_cnt; blk1++) {
#pragma omp parallel for
        for (int k=0; k<this->nfreq; k++) {
          for (size_t mat_indx=0; mat_indx< npos; mat_indx++) {
            size_t interac_blk1 = blk1*npos+mat_indx;
            size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
            size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
            size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
            real_t** IN = &IN_[BLOCK_SIZE*interac_blk1];
            real_t** OUT= &OUT_[BLOCK_SIZE*interac_blk1];
            real_t* M = &matrix_M2L[mat_indx][k*2*NCHILD*NCHILD]; // k-th freq's (row) offset in matrix_M2L[mat_indx]
            for (size_t j=0; j<interac_cnt; j+=2) {
              real_t* M_   = M;
              real_t* IN0  = IN [j+0] + k*NCHILD*2;   // go to k-th freq chunk
              real_t* IN1  = IN [j+1] + k*NCHILD*2;
              real_t* OUT0 = OUT[j+0] + k*NCHILD*2;
              real_t* OUT1 = OUT[j+1] + k*NCHILD*2;
              matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
            }
          }
        }
      }
      // add flop
      add_flop((long long)(8*8*8)*(interaction_offset_f.size()/2)*this->nfreq);
    }
    
    void fft_up_equiv(std::vector<size_t>& fft_offset,
                      RealVec& all_up_equiv, AlignedVec& fft_in) {}
    
    void ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal,
                       AlignedVec& fft_out, RealVec& all_dn_equiv) {}

    void M2L(Nodes<T>& nodes) {
      int& nsurf_ = this->nsurf;
      size_t fft_size = 2 * NCHILD * this->nfreq;
      int nnodes = nodes.size();

      // allocate memory
      std::vector<T> all_up_equiv, all_dn_equiv;
      all_up_equiv.reserve(nnodes*nsurf_);   // use reserve() to avoid the overhead of calling constructor
      all_dn_equiv.reserve(nnodes*nsurf_);   // use pointer instead of iterator to access elements 
      AlignedVec fft_in, fft_out;
      fft_in.reserve(m2ldata.fft_offset.size()*fft_size);
      fft_out.reserve(m2ldata.ifft_offset.size()*fft_size);

      // gather all upward equivalent charges
#pragma omp parallel for collapse(2)
      for (int i=0; i<nnodes; i++) {
        for (int j=0; j<nsurf_; j++) {
          all_up_equiv[i*nsurf_+j] = nodes[i].up_equiv[j];
          all_dn_equiv[i*nsurf_+j] = nodes[i].dn_equiv[j];
        }
      }

      fft_up_equiv(m2ldata.fft_offset, all_up_equiv, fft_in);
      hadamard_product(m2ldata.interaction_count_offset, m2ldata.interaction_offset_f, fft_in, fft_out);
      ifft_dn_check(m2ldata.ifft_offset, m2ldata.ifft_scale, fft_out, all_dn_equiv);

      // scatter all downward check potentials
#pragma omp parallel for collapse(2)
      for (int i=0; i<nnodes; i++) {
        for (int j=0; j<nsurf_; j++) {
          nodes[i].dn_equiv[j] = all_dn_equiv[i*nsurf_+j];
        }
      }
    }
  };

  
  /** Below are member function specializations
   */
  template <>
  void FmmScaleInvariant<real_t>::precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    int& nsurf_ = this->nsurf;

    // compute kernel matrix
    RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
    RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
    RealVec matrix_c2e(nsurf_*nsurf_);  // UC2UE
    this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

    // svd
    RealVec S(nsurf_*nsurf_);  // singular values 
    RealVec U(nsurf_*nsurf_), VH(nsurf_*nsurf_);
    svd(nsurf_, nsurf_, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

    // pseudo-inverse
    real_t max_S = 0;
    for (int i=0; i<nsurf_; i++) {
      max_S = fabs(S[i*nsurf_+i])>max_S ? fabs(S[i*nsurf_+i]) : max_S;
    }
    for (int i=0; i<nsurf_; i++) {
      S[i*nsurf_+i] = S[i*nsurf_+i]>EPS*max_S*4 ? 1.0/S[i*nsurf_+i] : 0.0;
    }
    RealVec V = transpose(VH, nsurf_, nsurf_);
    matrix_UC2E_U = transpose(U, nsurf_, nsurf_);
    gemm(nsurf_, nsurf_, nsurf_, &V[0], &S[0], &matrix_UC2E_V[0]);
    matrix_DC2E_U = VH;
    gemm(nsurf_, nsurf_, nsurf_, &U[0], &S[0], &matrix_DC2E_V[0]);
  }

  template <>
  void FmmScaleInvariant<complex_t>::precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    int& nsurf_ = this->nsurf;

    // compute kernel matrix
    RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
    RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
    ComplexVec matrix_c2e(nsurf_*nsurf_);  // UC2UE
    this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

    // svd
    RealVec S(nsurf_*nsurf_);  // singular values 
    ComplexVec U(nsurf_*nsurf_), VH(nsurf_*nsurf_);
    svd(nsurf_, nsurf_, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

    // pseudo-inverse
    real_t max_S = 0;
    for (int i=0; i<nsurf_; i++) {
      max_S = fabs(S[i*nsurf_+i])>max_S ? fabs(S[i*nsurf_+i]) : max_S;
    }
    for (int i=0; i<nsurf_; i++) {
      S[i*nsurf_+i] = S[i*nsurf_+i]>EPS*max_S*4 ? 1.0/S[i*nsurf_+i] : 0.0;
    }
    ComplexVec S_(nsurf_*nsurf_);
    for (size_t i=0; i<S_.size(); i++) {   // convert S to complex type
      S_[i] = S[i];
    }
    ComplexVec V = conjugate_transpose(VH, nsurf_, nsurf_);
    ComplexVec UH = conjugate_transpose(U, nsurf_, nsurf_);
    matrix_UC2E_U = UH;
    gemm(nsurf_, nsurf_, nsurf_, &V[0], &S_[0], &matrix_UC2E_V[0]);
    matrix_DC2E_U = transpose(V, nsurf_, nsurf_);
    ComplexVec UHT = transpose(UH, nsurf_, nsurf_);
    gemm(nsurf_, nsurf_, nsurf_, &UHT[0], &S_[0], &matrix_DC2E_V[0]);
  }

  //! member function specialization for real type
  template <>
  void FmmScaleInvariant<real_t>::precompute_M2L() {
    int n1 = this->p * 2;
    int& nconv_ = this->nconv;
    int& nfreq_ = this->nfreq;
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*nfreq_));
    // create fft plan
    RealVec fftw_in(nconv_);
    RealVec fftw_out(2*nfreq_);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft_r2c(3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()), FFTW_ESTIMATE);
    // compute M2L kernel matrix, perform DFT
    RealVec trg_coord(3,0);
#pragma omp parallel for
    for (size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for (int d=0; d<3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 / 0.5;  // relative coords
      }
      RealVec conv_coord = convolution_grid(this->p, this->r0, 0, coord);   // convolution grid
      RealVec conv_value(nconv_);   // potentials on convolution grid
      this->kernel_matrix(conv_coord, trg_coord, conv_value);
      fft_execute_dft_r2c(plan, conv_value.data(), reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
    for (size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
      for (int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
        int child_rel_idx = M2L_INDEX_MAP[i][j];
        if (child_rel_idx != -1) {
          for (int k=0; k<nfreq_; k++) {   // loop over frequencies
            int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
            matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / nconv_;   // real
            matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / nconv_;   // imag
          }
        }
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  template <>
  void FmmScaleInvariant<real_t>::fft_up_equiv(std::vector<size_t>& fft_offset,
                                               RealVec& all_up_equiv, AlignedVec& fft_in) {
    int& nsurf_ = this->nsurf;
    int& nconv_ = this->nconv;
    int& nfreq_ = this->nfreq;
    int n1 = 2 * this->p;
    auto map = generate_surf2conv_up(this->p);

    size_t fft_size = 2 * NCHILD * nfreq_;
    AlignedVec fftw_in(nconv_ * NCHILD);
    AlignedVec fftw_out(fft_size);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                          (real_t*)&fftw_in[0], nullptr, 1, nconv_,
                                          (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq_,
                                          FFTW_ESTIMATE);
#pragma omp parallel for
    for (size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fft_size, 0);
      real_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's upward_equiv in all_up_equiv, size=8*nsurf_
      // upward_equiv_fft (input of r2c) here should have a size of N3*NCHILD
      // the node_idx's chunk of fft_out has a size of 2*N3_*NCHILD
      // since it's larger than what we need,  we can use fft_out as fftw_in buffer here
      real_t* up_equiv_f = &fft_in[fft_size*node_idx]; // offset ptr of node_idx in fft_in vector, size=fft_size
      std::memset(up_equiv_f, 0, fft_size*sizeof(real_t));  // initialize fft_in to 0
      for (int k=0; k<nsurf_; k++) {
        size_t idx = map[k];
        for (int j=0; j<NCHILD; j++)
          up_equiv_f[idx+j*nconv_] = up_equiv[j*nsurf_+k];
      }
      fft_execute_dft_r2c(plan, up_equiv_f, (fft_complex*)&buffer[0]);
      // add flop
      double add, mul, fma;
      fft_flops(plan, &add, &mul, &fma);
      add_flop((long long)(add + mul + 2*fma));
      for (int k=0; k<nfreq_; k++) {
        for (int j=0; j<NCHILD; j++) {
          up_equiv_f[2*(NCHILD*k+j)+0] = buffer[2*(nfreq_*j+k)+0];
          up_equiv_f[2*(NCHILD*k+j)+1] = buffer[2*(nfreq_*j+k)+1];
        }
      }
    }
    fft_destroy_plan(plan);
  }

  template <>
  void FmmScaleInvariant<real_t>::ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal,
                       AlignedVec& fft_out, RealVec& all_dn_equiv) {
    int& nsurf_ = this->nsurf;
    int& nconv_ = this->nconv;
    int& nfreq_ = this->nfreq;
    int n1 = 2 * this->p;
    auto map = generate_surf2conv_dn(this->p);

    size_t fft_size = 2 * NCHILD * nfreq_;
    AlignedVec fftw_in(fft_size);
    AlignedVec fftw_out(nconv_ * NCHILD);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                                 (fft_complex*)&fftw_in[0], nullptr, 1, nfreq_,
                                 (real_t*)(&fftw_out[0]), nullptr, 1, nconv_,
                                 FFTW_ESTIMATE);
#pragma omp parallel for
    for (size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fft_size, 0);
      RealVec buffer1(fft_size, 0);
      real_t* dn_check_f = &fft_out[fft_size*node_idx];  // offset ptr for node_idx in fft_out vector, size=fft_size
      real_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * nsurf_
      for (int k=0; k<nfreq_; k++)
        for (int j=0; j<NCHILD; j++) {
          buffer0[2*(nfreq_*j+k)+0] = dn_check_f[2*(NCHILD*k+j)+0];
          buffer0[2*(nfreq_*j+k)+1] = dn_check_f[2*(NCHILD*k+j)+1];
        }
      fft_execute_dft_c2r(plan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
      // add flop
      double add, mul, fma;
      fft_flops(plan, &add, &mul, &fma);
      add_flop((long long)(add + mul + 2*fma));
      for (int k=0; k<nsurf_; k++) {
        size_t idx = map[k];
        for (int j=0; j<NCHILD; j++)
          dn_equiv[nsurf_*j+k] += buffer1[idx+j*nconv_] * ifft_scal[node_idx];
      }
    }
    fft_destroy_plan(plan);
  }
}  // end namespace
#endif
