#ifndef fmm_h
#define fmm_h
#include <cstring>      // std::memset
#include <fstream>      // std::ofstream
#include <type_traits>  // std::is_same
#include "fmm_base.h"
#include "intrinsics.h"
#include "math_wrapper.h"

namespace exafmm_t {
  template <typename T>
  class Fmm : public FmmBase<T> {
  public:
    /* precomputation matrices */
    std::vector<std::vector<T>> matrix_UC2E_U;
    std::vector<std::vector<T>> matrix_UC2E_V;
    std::vector<std::vector<T>> matrix_DC2E_U;
    std::vector<std::vector<T>> matrix_DC2E_V;
    std::vector<std::vector<std::vector<T>>> matrix_M2M;
    std::vector<std::vector<std::vector<T>>> matrix_L2L;

    std::vector<M2LData> m2ldata;

    /* constructors */
    Fmm() {}
    Fmm(int p_, int ncrit_, int depth_) : FmmBase<T>(p_, ncrit_, depth_) {}

    /* precomputation */
    //! Setup the sizes of precomputation matrices
    void initialize_matrix() {
      int nsurf = this->nsurf;
      int depth = this->depth;
      matrix_UC2E_V.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_UC2E_U.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_DC2E_V.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_DC2E_U.resize(depth+1, std::vector<T>(nsurf*nsurf));
      matrix_M2M.resize(depth+1);
      matrix_L2L.resize(depth+1);
      for (int level=0; level<=depth; ++level) {
        matrix_M2M[level].resize(REL_COORD[M2M_Type].size(), std::vector<T>(nsurf*nsurf));
        matrix_L2L[level].resize(REL_COORD[L2L_Type].size(), std::vector<T>(nsurf*nsurf));
      }
    }

    //! Precompute M2M and L2L
    void precompute_M2M() {
      int nsurf = this->nsurf;
      real_t parent_coord[3] = {0, 0, 0};
      for (int level=0; level<=this->depth; level++) {
        RealVec parent_up_check_surf = surface(this->p, this->r0, level, parent_coord, 2.95);
        real_t s = this->r0 * powf(0.5, level+1);
        int npos = REL_COORD[M2M_Type].size();  // number of relative positions
#pragma omp parallel for
        for(int i=0; i<npos; i++) {
          // compute kernel matrix
          ivec3& coord = REL_COORD[M2M_Type][i];
          real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                   parent_coord[1] + coord[1]*s,
                                   parent_coord[2] + coord[2]*s};
          RealVec child_up_equiv_surf = surface(this->p, this->r0, level+1, child_coord, 1.05);
          std::vector<T> matrix_pc2ce(nsurf*nsurf);
          this->kernel_matrix(parent_up_check_surf, child_up_equiv_surf, matrix_pc2ce);
          // M2M
          std::vector<T> buffer(nsurf*nsurf);
          gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_U[level][0]), &matrix_pc2ce[0], &buffer[0]);
          gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &(matrix_M2M[level][i][0]));
          // L2L
          matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
          gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &(matrix_DC2E_V[level][0]), &buffer[0]);
          gemm(nsurf, nsurf, nsurf, &buffer[0], &(matrix_DC2E_U[level][0]), &(matrix_L2L[level][i][0]));
        }
      }
    }

    //! Precompute UC2UE and DC2DE matrices
    void precompute_check2equiv() {}

    //! Precompute M2L
    void precompute_M2L(std::ofstream& file) {}

    //! Save precomputation matrices
    void save_matrix(std::ofstream& file) {
      file.write(reinterpret_cast<char*>(&this->r0), sizeof(real_t));  // r0
      size_t size = this->nsurf * this->nsurf;
      for(int l=0; l<=this->depth; l++) {
        // UC2E, DC2E
        file.write(reinterpret_cast<char*>(&matrix_UC2E_U[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_UC2E_V[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_DC2E_U[l][0]), size*sizeof(T));
        file.write(reinterpret_cast<char*>(&matrix_DC2E_V[l][0]), size*sizeof(T));
        // M2M, L2L
        for (auto & vec : matrix_M2M[l]) {
          file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
        }
        for (auto & vec : matrix_L2L[l]) {
          file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
        }
      }
    }

    //! Check and load precomputation matrices
    void load_matrix() {
      int nsurf = this->nsurf;
      int depth = this->depth;
      size_t size_M2L = this->nfreq * 2 * NCHILD * NCHILD;
      size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * nsurf * nsurf * (depth+1) * sizeof(T) 
                       + REL_COORD[M2L_Type].size() * size_M2L * depth * sizeof(real_t)
                       + 1 * sizeof(real_t);   // +1 denotes r0
      std::ifstream file(this->filename, std::ifstream::binary);
      if (file.good()) {
        file.seekg(0, file.end);
        if (size_t(file.tellg()) == file_size) {   // if file size is correct
          file.seekg(0, file.beg);  // move the position back to the beginning
          real_t r0_;
          file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
          if (this->r0 == r0_) {    // if radius match
            size_t size = nsurf * nsurf;
            for (int l=0; l<depth; l++) {
              // UC2E, DC2E
              file.read(reinterpret_cast<char*>(&matrix_UC2E_U[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_UC2E_V[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_DC2E_U[l][0]), size*sizeof(T));
              file.read(reinterpret_cast<char*>(&matrix_DC2E_V[l][0]), size*sizeof(T));
              // M2M, L2L
              for (auto& vec : matrix_M2M[l]) {
                file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
              }
              for (auto& vec : matrix_L2L[l]) {
                file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(T));
              }
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
        std::remove(this->filename.c_str());
        std::ofstream file(this->filename, std::ofstream::binary);
        save_matrix(file);
        precompute_M2L(file);
        file.close();
      }
    }

    //! P2M operator
    void P2M(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> up_check_surf;
      up_check_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        up_check_surf[level].resize(nsurf*3);
        up_check_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // calculate upward check potential induced by sources' charges
        RealVec check_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
          check_coord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
          check_coord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
          check_coord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
        }
        this->potential_P2P(leaf->src_coord, leaf->src_value,
                            check_coord, leaf->up_equiv);
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_UC2E_U[level][0]), &(leaf->up_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->up_equiv[k] = equiv[k];
      }
    }

    //! L2P operator
    void L2P(NodePtrs<T>& leafs) {
      int nsurf = this->nsurf;
      real_t c[3] = {0,0,0};
      std::vector<RealVec> dn_equiv_surf;
      dn_equiv_surf.resize(this->depth+1);
      for (int level=0; level<=this->depth; level++) {
        dn_equiv_surf[level].resize(nsurf*3);
        dn_equiv_surf[level] = surface(this->p, this->r0, level, c, 2.95);
      }
#pragma omp parallel for
      for (size_t i=0; i<leafs.size(); i++) {
        Node<T>* leaf = leafs[i];
        int level = leaf->level;
        // down check surface potential -> equivalent surface charge
        std::vector<T> buffer(nsurf);
        std::vector<T> equiv(nsurf);
        gemv(nsurf, nsurf, &(matrix_DC2E_U[level][0]), &(leaf->dn_equiv[0]), &buffer[0]);
        gemv(nsurf, nsurf, &(matrix_DC2E_V[level][0]), &buffer[0], &equiv[0]);
        for (int k=0; k<nsurf; k++)
          leaf->dn_equiv[k] = equiv[k];
        // equivalent surface charge -> target potential
        RealVec equiv_coord(nsurf*3);
        for (int k=0; k<nsurf; k++) {
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
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant])
#pragma omp task untied
          M2M(node->children[octant]);
      }
#pragma omp taskwait
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_M2M[level][octant][0]), &child->up_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++) {
            node->up_equiv[k] += buffer[k];
          }
        }
      }
    }
  
    //! L2L operator
    void L2L(Node<T>* node) {
      int nsurf = this->nsurf;
      if (node->is_leaf) return;
      for (int octant=0; octant<8; octant++) {
        if (node->children[octant]) {
          Node<T>* child = node->children[octant];
          std::vector<T> buffer(nsurf);
          int level = node->level;
          gemv(nsurf, nsurf, &(matrix_L2L[level][octant][0]), &node->dn_equiv[0], &buffer[0]);
          for (int k=0; k<nsurf; k++)
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

    void M2L_setup(NodePtrs<T>& nonleafs) {
      int nsurf = this->nsurf;
      int depth = this->depth;
      int npos = REL_COORD[M2L_Type].size();  // number of M2L relative positions
      m2ldata.resize(depth);                  // initialize m2ldata

      // construct lists of target nodes for M2L operator at each level
      std::vector<NodePtrs<T>> trg_nodes(depth);
      for (size_t i=0; i<nonleafs.size(); i++) {
        trg_nodes[nonleafs[i]->level].push_back(nonleafs[i]);
      }

      // prepare for m2ldata for each level
      for (int l=0; l<depth; l++) {
        // construct M2L source nodes for current level
        std::set<Node<T>*> src_nodes_;
        for (size_t i=0; i<trg_nodes[l].size(); i++) {
          NodePtrs<T>& M2L_list = trg_nodes[l][i]->M2L_list;
          for (size_t k=0; k<npos; k++) {
            if (M2L_list[k])
              src_nodes_.insert(M2L_list[k]);
          }
        }
        NodePtrs<T> src_nodes;
        auto it = src_nodes_.begin();
        for (; it!=src_nodes_.end(); it++) {
          src_nodes.push_back(*it);
        }
        // prepare the indices of src_nodes & trg_nodes in all_up_equiv & all_dn_equiv
        std::vector<size_t> fft_offset(src_nodes.size());       // displacement in all_up_equiv
        std::vector<size_t> ifft_offset(trg_nodes[l].size());  // displacement in all_dn_equiv
        for (size_t i=0; i<src_nodes.size(); i++) {
          fft_offset[i] = src_nodes[i]->children[0]->idx * nsurf;
        }
        for (size_t i=0; i<trg_nodes[l].size(); i++) {
          ifft_offset[i] = trg_nodes[l][i]->children[0]->idx * nsurf;
        }

        // calculate interaction_offset_f & interaction_count_offset
        std::vector<size_t> interaction_offset_f;
        std::vector<size_t> interaction_count_offset;
        for (size_t i=0; i<src_nodes.size(); i++) {
          src_nodes[i]->idx_M2L = i;  // node_id: node's index in nodes_in list
        }
        size_t n_blk1 = trg_nodes[l].size() * sizeof(real_t) / CACHE_SIZE;
        if (n_blk1==0) n_blk1 = 1;
        size_t interaction_count_offset_ = 0;
        size_t fft_size = 2 * NCHILD * this->nfreq;
        for (size_t blk1=0; blk1<n_blk1; blk1++) {
          size_t blk1_start=(trg_nodes[l].size()* blk1   )/n_blk1;
          size_t blk1_end  =(trg_nodes[l].size()*(blk1+1))/n_blk1;
          for (size_t k=0; k<npos; k++) {
            for (size_t i=blk1_start; i<blk1_end; i++) {
              NodePtrs<T>& M2L_list = trg_nodes[l][i]->M2L_list;
              if (M2L_list[k]) {
                interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fft_size);   // node_in's displacement in fft_in
                interaction_offset_f.push_back(        i           * fft_size);   // node_out's displacement in fft_out
                interaction_count_offset_++;
              }
            }
            interaction_count_offset.push_back(interaction_count_offset_);
          }
        }
        m2ldata[l].fft_offset = fft_offset;
        m2ldata[l].ifft_offset = ifft_offset;
        m2ldata[l].interaction_offset_f = interaction_offset_f;
        m2ldata[l].interaction_count_offset = interaction_count_offset;
      }
    }

  void hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                        AlignedVec& fft_in, AlignedVec& fft_out, std::vector<AlignedVec>& matrix_M2L) {
      int p = this->p;
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
      for(size_t i=0; i<fft_out.capacity()/fft_size; ++i) {
        std::memset(fft_out.data()+i*fft_size, 0, fft_size*sizeof(real_t));
      }
      
      #pragma omp parallel for
      for(size_t interac_blk1=0; interac_blk1<blk1_cnt*npos; interac_blk1++) {
        size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
        size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
        size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
        for(size_t j=0; j<interac_cnt; j++) {
          IN_ [BLOCK_SIZE*interac_blk1 +j] = &fft_in[interaction_offset_f[(interaction_count_offset0+j)*2+0]];
          OUT_[BLOCK_SIZE*interac_blk1 +j] = &fft_out[interaction_offset_f[(interaction_count_offset0+j)*2+1]];
        }
        IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec0[0];
        OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec1[0];
      }

      for(size_t blk1=0; blk1<blk1_cnt; blk1++) {
      #pragma omp parallel for
        for(int k=0; k<this->nfreq; k++) {
          for(size_t mat_indx=0; mat_indx< npos; mat_indx++) {
            size_t interac_blk1 = blk1*npos+mat_indx;
            size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
            size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
            size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
            real_t** IN = &IN_[BLOCK_SIZE*interac_blk1];
            real_t** OUT= &OUT_[BLOCK_SIZE*interac_blk1];
            real_t* M = &matrix_M2L[mat_indx][k*2*NCHILD*NCHILD]; // k-th freq's (row) offset in matrix_M2L
            for(size_t j=0; j<interac_cnt; j+=2) {
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
    }

    void fft_up_equiv(std::vector<size_t>& fft_offset, std::vector<T>& all_up_equiv, AlignedVec& fft_in) {}

    void ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out, std::vector<T>& all_dn_equiv) {}

    void M2L(Nodes<T>& nodes) {
      int nsurf = this->nsurf;
      int nfreq = this->nfreq;
      int fft_size = 2 * NCHILD * nfreq;
      int nnodes = nodes.size();
      int npos = REL_COORD[M2L_Type].size();   // number of relative positions

      // allocate memory
      std::vector<T> all_up_equiv, all_dn_equiv;
      all_up_equiv.reserve(nnodes*nsurf);
      all_dn_equiv.reserve(nnodes*nsurf);
      std::vector<AlignedVec> matrix_M2L(npos, AlignedVec(fft_size*NCHILD, 0));

      // setup ifstream of M2L precomputation matrix
      std::ifstream ifile(this->filename, std::ifstream::binary);
      ifile.seekg(0, ifile.end);
      size_t fsize = ifile.tellg();   // file size in bytes
      size_t msize = NCHILD * NCHILD * nfreq * 2 * sizeof(real_t);   // size in bytes for each M2L matrix
      ifile.seekg(fsize - this->depth*npos*msize, ifile.beg);   // go to the start of M2L section
      
      // collect all upward equivalent charges
#pragma omp parallel for collapse(2)
      for (int i=0; i<nnodes; ++i) {
        for (int j=0; j<nsurf; ++j) {
          all_up_equiv[i*nsurf+j] = nodes[i].up_equiv[j];
          all_dn_equiv[i*nsurf+j] = nodes[i].dn_equiv[j];
        }
      }
      // FFT-accelerate M2L
      for (int l=0; l<this->depth; ++l) {
        // load M2L matrix for current level
        for (int i=0; i<npos; ++i) {
          ifile.read(reinterpret_cast<char*>(matrix_M2L[i].data()), msize);
        }
        AlignedVec fft_in, fft_out;
        fft_in.reserve(m2ldata[l].fft_offset.size()*fft_size);
        fft_out.reserve(m2ldata[l].ifft_offset.size()*fft_size);
        fft_up_equiv(m2ldata[l].fft_offset, all_up_equiv, fft_in);
        hadamard_product(m2ldata[l].interaction_count_offset, 
                         m2ldata[l].interaction_offset_f, 
                         fft_in, fft_out, matrix_M2L);
        ifft_dn_check(m2ldata[l].ifft_offset, fft_out, all_dn_equiv);
      }
      // update all downward check potentials
#pragma omp parallel for collapse(2)
      for (int i=0; i<nnodes; ++i) {
        for (int j=0; j<nsurf; ++j) {
          nodes[i].dn_equiv[j] = all_dn_equiv[i*nsurf+j];
        }
      }
      ifile.close();   // close ifstream
    }
  };
  
  /** Below are member function specializations
   */
  template <>
  void Fmm<real_t>::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;
#pragma omp parallel for
    for (int level=0; level<=this->depth; ++level) {
      // compute kernel matrix
      RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
      RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
      RealVec matrix_c2e(nsurf*nsurf);  // UC2UE
      this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

      // svd
      RealVec S(nsurf*nsurf);  // singular values 
      RealVec U(nsurf*nsurf), VT(nsurf*nsurf);
      svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VT[0]);

      // pseudo-inverse
      real_t max_S = 0;
      for (int i=0; i<nsurf; i++) {
        max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
      }
      for (int i=0; i<nsurf; i++) {
        S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
      }
      RealVec V = transpose(VT, nsurf, nsurf);
      matrix_UC2E_U[level] = transpose(U, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &(matrix_UC2E_V[level][0]));
      matrix_DC2E_U[level] = VT;
      gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &(matrix_DC2E_V[level][0]));
    }
  }

  template <>
  void Fmm<complex_t>::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    int nsurf = this->nsurf;
#pragma omp parallel for
    for (int level=0; level<=this->depth; ++level) {
      // compute kernel matrix
      RealVec up_check_surf = surface(this->p, this->r0, level, c, 2.95);
      RealVec up_equiv_surf = surface(this->p, this->r0, level, c, 1.05);
      ComplexVec matrix_c2e(nsurf*nsurf);  // UC2UE
      this->kernel_matrix(up_check_surf, up_equiv_surf, matrix_c2e);

      // svd
      RealVec S(nsurf*nsurf);  // singular values 
      ComplexVec U(nsurf*nsurf), VH(nsurf*nsurf);
      svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VH[0]);

      // pseudo-inverse
      real_t max_S = 0;
      for (int i=0; i<nsurf; i++) {
        max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
      }
      for (int i=0; i<nsurf; i++) {
        S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
      }
      ComplexVec S_(nsurf*nsurf);
      for (size_t i=0; i<S_.size(); i++) {   // convert S to complex type
        S_[i] = S[i];
      }
      ComplexVec V = conjugate_transpose(VH, nsurf, nsurf);
      ComplexVec UH = conjugate_transpose(U, nsurf, nsurf);
      matrix_UC2E_U[level] = UH;
      gemm(nsurf, nsurf, nsurf, &V[0], &S_[0], &(matrix_UC2E_V[level][0]));
      matrix_DC2E_U[level] = transpose(V, nsurf, nsurf);
      ComplexVec UHT = transpose(UH, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &UHT[0], &S_[0], &(matrix_DC2E_V[level][0]));
    }
  } 

  //! member function specialization for real type
  template <>
  void Fmm<real_t>::precompute_M2L(std::ofstream& file) {
    int n1 = this->p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    int fft_size = 2 * nfreq * NCHILD * NCHILD;
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*nfreq));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fft plan
    RealVec fftw_in(nconv);
    RealVec fftw_out(2*nfreq);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft_r2c(3, dim, fftw_in.data(), reinterpret_cast<fft_complex*>(fftw_out.data()), FFTW_ESTIMATE);
    RealVec trg_coord(3,0);
    for (int l=1; l<this->depth+1; ++l) {
      // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
        real_t coord[3];
        for (int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 * powf(0.5, l-1);  // relative coords
        }
        RealVec conv_coord = convolution_grid(this->p, this->r0, l, coord);   // convolution grid
        RealVec conv_value(nconv);   // potentials on convolution grid
        this->kernel_matrix(conv_coord, trg_coord, conv_value);
        fft_execute_dft_r2c(plan, conv_value.data(), reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
      }
      // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
        for (int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
          int child_rel_idx = M2L_INDEX_MAP[i][j];
          if (child_rel_idx != -1) {
            for (int k=0; k<nfreq; k++) {   // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / nconv;   // real
              matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / nconv;   // imag
            }
          }
        }
      }
      // write to file
      for(auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()), fft_size*sizeof(real_t));
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  //! member function specialization for complex type
  template <>
  void Fmm<complex_t>::precompute_M2L(std::ofstream& file) {
    int n1 = this->p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    int fft_size = 2 * nfreq * NCHILD * NCHILD;
    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*nfreq));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fft plan
    RealVec fftw_in(nconv);
    RealVec fftw_out(2*nfreq);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft(3, dim,
                                 reinterpret_cast<fft_complex*>(fftw_in.data()),
                                 reinterpret_cast<fft_complex*>(fftw_out.data()),
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    RealVec trg_coord(3,0);
    for (int l=1; l<this->depth+1; ++l) {
      // compute M2L kernel matrix, perform DFT
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
        real_t coord[3];
        for (int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * this->r0 * powf(0.5, l-1);  // relative coords
        }
        RealVec conv_coord = convolution_grid(this->p, this->r0, l, coord);   // convolution grid
        ComplexVec conv_value(nconv);   // potentials on convolution grid
        this->kernel_matrix(conv_coord, trg_coord, conv_value);
        fft_execute_dft(plan, reinterpret_cast<fft_complex*>(conv_value.data()),
                              reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
      }
      // convert M2L_Helper to M2L and reorder data layout to improve locality
#pragma omp parallel for
      for (size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
        for (int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
          int child_rel_idx = M2L_INDEX_MAP[i][j];
          if (child_rel_idx != -1) {
            for (int k=0; k<nfreq; k++) {   // loop over frequencies
              int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
              matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / nconv;   // real
              matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / nconv;   // imag
            }
          }
        }
      }
      // write to file
      for(auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()), fft_size*sizeof(real_t));
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  template <>
  void Fmm<real_t>::fft_up_equiv(std::vector<size_t>& fft_offset, RealVec& all_up_equiv, AlignedVec& fft_in) {
    int nsurf = this->nsurf;
    int p = this->p;
    int n1 = p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for (int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, this->r0, 0, c, (real_t)(p-1), true);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p-1-surf[i*3]+0.5))
             + ((size_t)(p-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(p-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fft_size = 2 * NCHILD * nfreq;
    AlignedVec fftw_in(nconv * NCHILD);
    AlignedVec fftw_out(fft_size);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                          (real_t*)&fftw_in[0], nullptr, 1, nconv,
                                          (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq, 
                                          FFTW_ESTIMATE);

#pragma omp parallel for
    for (size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fft_size, 0);
      RealVec equiv_t(NCHILD*nconv, 0.);

      real_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's up_equiv in all_up_equiv, size=8*nsurf
      real_t* up_equiv_f = &fft_in[fft_size*node_idx];   // offset ptr of node_idx in fft_in vector, size=fftsize

      for (int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for (int j0=0; j0<NCHILD; j0++)
          equiv_t[idx+j0*nconv] = up_equiv[j0*nsurf+k];
      }
      fft_execute_dft_r2c(plan, &equiv_t[0], (fft_complex*)&buffer[0]);
      for (int j=0; j<nfreq; j++) {
        for (int k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(nfreq*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(nfreq*k+j)+1];
        }
      }
    }
    fft_destroy_plan(plan);
  }

  template <>
  void Fmm<complex_t>::fft_up_equiv(std::vector<size_t>& fft_offset, ComplexVec& all_up_equiv, AlignedVec& fft_in) {
    int nsurf = this->nsurf;
    int p = this->p;
    int n1 = p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for (int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, this->r0, 0, c, (real_t)(p-1), true);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p-1-surf[i*3]+0.5))
             + ((size_t)(p-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(p-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fft_size = 2 * NCHILD * nfreq;
    ComplexVec fftw_in(nconv * NCHILD);
    AlignedVec fftw_out(fft_size);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft(3, dim, NCHILD, reinterpret_cast<fft_complex*>(&fftw_in[0]),
                                      nullptr, 1, nconv, (fft_complex*)(&fftw_out[0]), nullptr, 1, nfreq, 
                                      FFTW_FORWARD, FFTW_ESTIMATE);

#pragma omp parallel for
    for (size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fft_size, 0);
      ComplexVec equiv_t(NCHILD*nconv, complex_t(0.,0.));

      complex_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's up_equiv in all_up_equiv, size=8*nsurf
      real_t* up_equiv_f = &fft_in[fft_size*node_idx];   // offset ptr of node_idx in fft_in vector, size=fftsize

      for (int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for (int j0=0; j0<NCHILD; j0++)
          equiv_t[idx+j0*nconv] = up_equiv[j0*nsurf+k];
      }
      fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&equiv_t[0]), (fft_complex*)&buffer[0]);
      for (int j=0; j<nfreq; j++) {
        for (int k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(nfreq*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(nfreq*k+j)+1];
        }
      }
    }
    fft_destroy_plan(plan);
  }

  template <>
  void Fmm<real_t>::ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out, RealVec& all_dn_equiv) {
    int nsurf = this->nsurf;
    int p = this->p;
    int n1 = p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for (int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, this->r0, 0, c, (real_t)(p-1), true);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p*2-0.5-surf[i*3]))
             + ((size_t)(p*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(p*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fft_size = 2 * NCHILD * nfreq;
    AlignedVec fftw_in(fft_size);
    AlignedVec fftw_out(nconv * NCHILD);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                    (fft_complex*)(&fftw_in[0]), nullptr, 1, nfreq, 
                    (real_t*)(&fftw_out[0]), nullptr, 1, nconv, 
                    FFTW_ESTIMATE);

#pragma omp parallel for
    for (size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fft_size, 0);
      RealVec buffer1(fft_size, 0);
      real_t* dn_check_f = &fft_out[fft_size*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      real_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * nsurf
      for (int j=0; j<nfreq; j++)
        for (int k=0; k<NCHILD; k++) {
          buffer0[2*(nfreq*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(nfreq*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(plan, (fft_complex*)&buffer0[0], (real_t*)(&buffer1[0]));
      for (int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for (int j0=0; j0<NCHILD; j0++)
          dn_equiv[nsurf*j0+k] += buffer1[idx+j0*nconv];
      }
    }
    fft_destroy_plan(plan);
  }
  
  template <>
  void Fmm<complex_t>::ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out, ComplexVec& all_dn_equiv) {
    int nsurf = this->nsurf;
    int p = this->p;
    int n1 = p * 2;
    int nconv = this->nconv;
    int nfreq = this->nfreq;
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for (int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, this->r0, 0, c, (real_t)(p-1), true);
    for (size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p*2-0.5-surf[i*3]))
             + ((size_t)(p*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(p*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fft_size = 2 * NCHILD * nfreq;
    AlignedVec fftw_in(fft_size);
    ComplexVec fftw_out(nconv*NCHILD);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft(3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1, nfreq, 
                                      reinterpret_cast<fft_complex*>(&fftw_out[0]), nullptr, 1, nconv, 
                                      FFTW_BACKWARD, FFTW_ESTIMATE);

#pragma omp parallel for
    for (size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fft_size, 0);
      ComplexVec buffer1(NCHILD*nconv, 0);
      real_t* dn_check_f = &fft_out[fft_size*node_idx];
      complex_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];
      for (int j=0; j<nfreq; j++)
        for (int k=0; k<NCHILD; k++) {
          buffer0[2*(nfreq*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(nfreq*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft(plan, (fft_complex*)&buffer0[0], reinterpret_cast<fft_complex*>(&buffer1[0]));
      for (int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for (int j0=0; j0<NCHILD; j0++)
          dn_equiv[nsurf*j0+k]+=buffer1[idx+j0*nconv];
      }
    }
    fft_destroy_plan(plan);
  }
}  // end namespace
#endif