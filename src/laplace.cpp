#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "laplace.h"
#include "math_wrapper.h"

namespace exafmm_t {
  bool LaplaceFMM::load_matrix() {
    std::ifstream file(filename, std::ifstream::binary);
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * nsurf * nsurf
                     +  REL_COORD[M2L_Type].size() * fft_size + 1;   // +1 denotes r0
    file_size *= sizeof(real_t);
    if (file.good()) {     // if file exists
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == file_size) {   // if file size is correct
        file.seekg(0, file.beg);         // move the position back to the beginning
        // check whether r0 matches
        real_t r0_;
        file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
        if (r0 != r0_) {
          return false;
        }
        size_t size = nsurf * nsurf;
        // UC2E, DC2E
        file.read(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(real_t));
        file.read(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(real_t));
        // M2M, L2L
        for(auto & vec : matrix_M2M) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
        }
        for(auto & vec : matrix_L2L) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
        }
        // M2L
        size = n3_ * 2 * NCHILD * NCHILD;
        for(auto & vec : matrix_M2L) {
          file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
        }
        file.close();
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  void LaplaceFMM::precompute() {
    // if matrix binary file exists
    filename = "laplace";
    filename += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(p);
    filename += ".dat";
    initialize_matrix();
    if (load_matrix()) {
      is_precomputed = false;
      return;
    } else {
      precompute_check2equiv();
      precompute_M2M();
      auto parent2child = map_matrix_index();
      precompute_M2L(parent2child);
      save_matrix();
    }
  }
  void LaplaceFMM::P2M(NodePtrs_t& leafs) {
    real_t c[3] = {0,0,0};
    std::vector<RealVec> up_check_surf;
    up_check_surf.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      up_check_surf[level].resize(nsurf*3);
      up_check_surf[level] = surface(p, r0, level, c, 2.95);
    }
    #pragma omp parallel for
    for(size_t i=0; i<leafs.size(); i++) {
      Node_t* leaf = leafs[i];
      int level = leaf->level;
      real_t scal = pow(0.5, level);  // scaling factor of UC2UE precomputation matrix
      // calculate upward check potential induced by sources' charges
      RealVec checkCoord(nsurf*3);
      for(int k=0; k<nsurf; k++) {
        checkCoord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
        checkCoord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
        checkCoord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
      }
      potential_P2P(leaf->src_coord, leaf->src_value, checkCoord, leaf->up_equiv);
      // convert upward check potential to upward equivalent charge
      RealVec buffer(nsurf);
      RealVec equiv(nsurf);
      gemv(nsurf, nsurf, &matrix_UC2E_U[0], &(leaf->up_equiv[0]), &buffer[0]);
      gemv(nsurf, nsurf, &matrix_UC2E_V[0], &buffer[0], &equiv[0]);
      // scale the check-to-equivalent conversion (precomputation)
      for(int k=0; k<nsurf; k++)
        leaf->up_equiv[k] = scal * equiv[k];
    }
  }

  void LaplaceFMM::M2M(Node_t* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant])
        #pragma omp task untied
        M2M(node->children[octant]);
    }
    #pragma omp taskwait
    // evaluate parent's upward equivalent charge from child's upward equivalent charge
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node_t* child = node->children[octant];
        RealVec buffer(nsurf);
        gemv(nsurf, nsurf, &(matrix_M2M[octant][0]), &child->up_equiv[0], &buffer[0]);
        for(int k=0; k<nsurf; k++) {
          node->up_equiv[k] += buffer[k];
        }
      }
    }
  }

  void LaplaceFMM::L2L(Node_t* node) {
    if(node->is_leaf) return;
    // evaluate child's downward check potential from parent's downward check potential
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node_t* child = node->children[octant];
        RealVec buffer(nsurf);
        gemv(nsurf, nsurf, &(matrix_L2L[octant][0]), &node->dn_equiv[0], &buffer[0]);
        for(int k=0; k<nsurf; k++)
          child->dn_equiv[k] += buffer[k];
      }
    }
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant])
        #pragma omp task untied
        L2L(node->children[octant]);
    }
    #pragma omp taskwait
  }

  void LaplaceFMM::L2P(NodePtrs_t& leafs) {
    real_t c[3] = {0.0};
    std::vector<RealVec> dn_equiv_surf;
    dn_equiv_surf.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      dn_equiv_surf[level].resize(nsurf*3);
      dn_equiv_surf[level] = surface(p, r0, level, c, 2.95);
    }
    #pragma omp parallel for
    for(size_t i=0; i<leafs.size(); i++) {
      Node_t* leaf = leafs[i];
      int level = leaf->level;
      real_t scal = pow(0.5, level);
      // convert downward check potential to downward equivalent charge
      RealVec buffer(nsurf);
      RealVec equiv(nsurf);
      gemv(nsurf, nsurf, &matrix_DC2E_U[0], &(leaf->dn_equiv[0]), &buffer[0]);
      gemv(nsurf, nsurf, &matrix_DC2E_V[0], &buffer[0], &equiv[0]);
      // scale the check-to-equivalent conversion (precomputation)
      for(int k=0; k<nsurf; k++)
        leaf->dn_equiv[k] = scal * equiv[k];
      // calculate targets' potential & gradient induced by downward equivalent charge
      RealVec equivCoord(nsurf*3);
      for(int k=0; k<nsurf; k++) {
        equivCoord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->x[0];
        equivCoord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->x[1];
        equivCoord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->x[2];
      }
      gradient_P2P(equivCoord, leaf->dn_equiv, leaf->trg_coord, leaf->trg_value);
    }
  }

  void LaplaceFMM::M2L_setup(NodePtrs_t& nonleafs) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t mat_cnt = REL_COORD[M2L_Type].size();
    // construct nodes_out & nodes_in
    NodePtrs_t& nodes_out = nonleafs;
    std::set<Node_t*> nodes_in_;
    for(size_t i=0; i<nodes_out.size(); i++) {
      NodePtrs_t& M2L_list = nodes_out[i]->M2L_list;
      for(size_t k=0; k<mat_cnt; k++) {
        if(M2L_list[k])
          nodes_in_.insert(M2L_list[k]);
      }
    }
    NodePtrs_t nodes_in;
    for(std::set<Node_t*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
      nodes_in.push_back(*node);
    }
    // prepare fft displ & fft scal
    std::vector<size_t> fft_offset(nodes_in.size());
    std::vector<size_t> ifft_offset(nodes_out.size());
    RealVec ifft_scale(nodes_out.size());
    for(size_t i=0; i<nodes_in.size(); i++) {
      fft_offset[i] = nodes_in[i]->children[0]->idx * nsurf;
    }
    for(size_t i=0; i<nodes_out.size(); i++) {
      int level = nodes_out[i]->level+1;
      ifft_offset[i] = nodes_out[i]->children[0]->idx * nsurf;
      ifft_scale[i] = powf(2.0, level);
    }
    // calculate interaction_offset_f & interaction_count_offset
    std::vector<size_t> interaction_offset_f;
    std::vector<size_t> interaction_count_offset;
    for(size_t i=0; i<nodes_in.size(); i++) {
     nodes_in[i]->idx_M2L = i;
    }
    size_t n_blk1 = nodes_out.size() * sizeof(real_t) / CACHE_SIZE;
    if(n_blk1==0) n_blk1 = 1;
    size_t interaction_count_offset_ = 0;
    size_t fftsize = 2 * 8 * n3_;
    for(size_t blk1=0; blk1<n_blk1; blk1++) {
      size_t blk1_start=(nodes_out.size()* blk1   )/n_blk1;
      size_t blk1_end  =(nodes_out.size()*(blk1+1))/n_blk1;
      for(size_t k=0; k<mat_cnt; k++) {
        for(size_t i=blk1_start; i<blk1_end; i++) {
          NodePtrs_t& M2L_list = nodes_out[i]->M2L_list;
          if(M2L_list[k]) {
            interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fftsize);   // node_in dspl
            interaction_offset_f.push_back(        i           * fftsize);   // node_out dspl
            interaction_count_offset_++;
          }
        }
        interaction_count_offset.push_back(interaction_count_offset_);
      }
    }
    m2ldata.fft_offset     = fft_offset;
    m2ldata.ifft_offset    = ifft_offset;
    m2ldata.ifft_scale    = ifft_scale;
    m2ldata.interaction_offset_f = interaction_offset_f;
    m2ldata.interaction_count_offset = interaction_count_offset;
  }

  void LaplaceFMM::hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                       AlignedVec& fft_in, AlignedVec& fft_out) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fftsize = 2 * 8 * n3_;
    AlignedVec zero_vec0(fftsize, 0.);
    AlignedVec zero_vec1(fftsize, 0.);

    size_t mat_cnt = matrix_M2L.size();
    size_t blk1_cnt = interaction_count_offset.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
    std::vector<real_t*> IN_(BLOCK_SIZE*blk1_cnt*mat_cnt);
    std::vector<real_t*> OUT_(BLOCK_SIZE*blk1_cnt*mat_cnt);

    // initialize fft_out with zero
    #pragma omp parallel for
    for(size_t i=0; i<fft_out.capacity()/fftsize; ++i) {
      std::memset(fft_out.data()+i*fftsize, 0, fftsize*sizeof(real_t));
    }

    #pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++) {
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
      for(int k=0; k<n3_; k++) {
        for(size_t mat_indx=0; mat_indx< mat_cnt; mat_indx++) {
          size_t interac_blk1 = blk1*mat_cnt+mat_indx;
          size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
          size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
          size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
          real_t** IN = &IN_[BLOCK_SIZE*interac_blk1];
          real_t** OUT= &OUT_[BLOCK_SIZE*interac_blk1];
          real_t* M = &matrix_M2L[mat_indx][k*2*NCHILD*NCHILD]; // k-th freq's (row) offset in matrix_M2L[mat_indx]
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

  void LaplaceFMM::fft_up_equiv(std::vector<size_t>& fft_offset,
                   RealVec& all_up_equiv, AlignedVec& fft_in) {
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for(int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, r0, 0, c, (real_t)(p-1), true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p-1-surf[i*3]+0.5))
             + ((size_t)(p-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(p-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(n3 * NCHILD);
    AlignedVec fftw_out(fftsize);
    int dim[3] = {2*p, 2*p, 2*p};
    fft_plan m2l_list_fftplan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                (real_t*)&fftw_in[0], nullptr, 1, n3,
                                (fft_complex*)(&fftw_out[0]), nullptr, 1, n3_,
                                FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fftsize, 0);
      real_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's upward_equiv in all_up_equiv, size=8*nsurf
      // upward_equiv_fft (input of r2c) here should have a size of N3*NCHILD
      // the node_idx's chunk of fft_out has a size of 2*N3_*NCHILD
      // since it's larger than what we need,  we can use fft_out as fftw_in buffer here
      real_t* up_equiv_f = &fft_in[fftsize*node_idx]; // offset ptr of node_idx in fft_in vector, size=fftsize
      std::memset(up_equiv_f, 0, fftsize*sizeof(real_t));  // initialize fft_in to 0
      for(int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<(int)NCHILD; j0++)
          up_equiv_f[idx+j0*n3] = up_equiv[j0*nsurf+k];
      }
      fft_execute_dft_r2c(m2l_list_fftplan, up_equiv_f, (fft_complex*)&buffer[0]);
      for(int j=0; j<n3_; j++) {
        for(size_t k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(n3_*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(n3_*k+j)+1];
        }
      }
    }
    fft_destroy_plan(m2l_list_fftplan);
  }

  void LaplaceFMM::ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal,
                       AlignedVec& fft_out, RealVec& all_dn_equiv) {
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(nsurf);
    real_t c[3]= {0.5, 0.5, 0.5};
    for(int d=0; d<3; d++) c[d] += 0.5*(p-2);
    RealVec surf = surface(p, r0, 0, c, (real_t)(p-1), true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(p*2-0.5-surf[i*3]))
             + ((size_t)(p*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(p*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(fftsize);
    AlignedVec fftw_out(n3 * NCHILD);
    int dim[3] = {2*p, 2*p, 2*p};
    fft_plan m2l_list_ifftplan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                                 (fft_complex*)&fftw_in[0], nullptr, 1, n3_,
                                 (real_t*)(&fftw_out[0]), nullptr, 1, n3,
                                 FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fftsize, 0);
      RealVec buffer1(fftsize, 0);
      real_t* dn_check_f = &fft_out[fftsize*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      real_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * nsurf
      for(int j=0; j<n3_; j++)
        for(size_t k=0; k<NCHILD; k++) {
          buffer0[2*(n3_*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(n3_*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
      for(int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dn_equiv[nsurf*j0+k] += buffer1[idx+j0*n3] * ifft_scal[node_idx];
      }
    }
    fft_destroy_plan(m2l_list_ifftplan);
  }

  void LaplaceFMM::M2L(Nodes_t& nodes) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fftsize = 2 * 8 * n3_;
    size_t numNodes = nodes.size();

    // allocate memory
    RealVec all_up_equiv, all_dn_equiv;
    all_up_equiv.reserve(numNodes*nsurf);   // use reserve() to avoid the overhead of calling constructor
    all_dn_equiv.reserve(numNodes*nsurf);   // use pointer instead of iterator to access elements 
    AlignedVec fft_in, fft_out;
    fft_in.reserve(m2ldata.fft_offset.size()*fftsize);
    fft_out.reserve(m2ldata.ifft_offset.size()*fftsize);

    // gather all upward equivalent charges
    #pragma omp parallel for collapse(2)
    for(size_t i=0; i<numNodes; i++) {
      for(int j=0; j<nsurf; j++) {
        all_up_equiv[i*nsurf+j] = nodes[i].up_equiv[j];
        all_dn_equiv[i*nsurf+j] = nodes[i].dn_equiv[j];
      }
    }

    fft_up_equiv(m2ldata.fft_offset, all_up_equiv, fft_in);
    hadamard_product(m2ldata.interaction_count_offset, m2ldata.interaction_offset_f, fft_in, fft_out);
    ifft_dn_check(m2ldata.ifft_offset, m2ldata.ifft_scale, fft_out, all_dn_equiv);

    // scatter all downward check potentials
    #pragma omp parallel for collapse(2)
    for(size_t i=0; i<numNodes; i++) {
      for(int j=0; j<nsurf; j++) {
        nodes[i].dn_equiv[j] = all_dn_equiv[i*nsurf+j];
      }
    }
  }





  RealVec LaplaceFMM::verify(NodePtrs_t& leafs) {
    int ntrgs = 10;
    int stride = leafs.size() / ntrgs;
    Nodes_t targets;
    for(int i=0; i<ntrgs; i++) {
      targets.push_back(*(leafs[i*stride]));
    }
    Nodes_t targets2 = targets;    // used for direct summation
#pragma omp parallel for
    for(size_t i=0; i<targets2.size(); i++) {
      Node_t* target = &targets2[i];
      std::fill(target->trg_value.begin(), target->trg_value.end(), 0.);
      for(size_t j=0; j<leafs.size(); j++) {
        gradient_P2P(leafs[j]->src_coord, leafs[j]->src_value, target->trg_coord, target->trg_value);
      }
    }
    real_t p_diff = 0, p_norm = 0, F_diff = 0, F_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      if (targets2[i].ntrgs != 0) {  // if current leaf is not empty
        p_norm += std::norm(targets2[i].trg_value[0]);
        p_diff += std::norm(targets2[i].trg_value[0] - targets[i].trg_value[0]);
        for(int d=1; d<4; d++) {
          F_diff += std::norm(targets2[i].trg_value[d] - targets[i].trg_value[d]);
          F_norm += std::norm(targets2[i].trg_value[d]);
        }
      }
    }
    RealVec rel_error(2);
    rel_error[0] = sqrt(p_diff/p_norm);   // potential error
    rel_error[1] = sqrt(F_diff/F_norm);   // gradient error

    return rel_error;
  }
}  // end namespace exafmm_t
