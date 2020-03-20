#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "modified_helmholtz.h"
#include "math_wrapper.h"

namespace exafmm_t {


  void ModifiedHelmholtzFMM::M2L_setup(NodePtrs_t& nonleafs) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t mat_cnt = REL_COORD[M2L_Type].size();
    // initialize m2ldata
    m2ldata.resize(depth);
    // construct M2L target nodes for each level
    std::vector<NodePtrs_t> nodes_out(depth);
    for(size_t i=0; i<nonleafs.size(); i++) {
      nodes_out[nonleafs[i]->level].push_back(nonleafs[i]);
    }
    // prepare for m2ldata for each level
    for(int l=0; l<depth; l++) {
      // construct M2L source nodes for current level
      std::set<Node_t*> nodes_in_;
      for(size_t i=0; i<nodes_out[l].size(); i++) {
        NodePtrs_t& M2L_list = nodes_out[l][i]->M2L_list;
        for(size_t k=0; k<mat_cnt; k++) {
          if(M2L_list[k])
            nodes_in_.insert(M2L_list[k]);
        }
      }
      NodePtrs_t nodes_in;
      for(std::set<Node_t*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
        nodes_in.push_back(*node);
      }
      // prepare fft displ
      std::vector<size_t> fft_offset(nodes_in.size());       // displacement in all_up_equiv
      std::vector<size_t> ifft_offset(nodes_out[l].size());  // displacement in all_dn_equiv
      for(size_t i=0; i<nodes_in.size(); i++) {
        fft_offset[i] = nodes_in[i]->children[0]->idx * nsurf;
      }
      for(size_t i=0; i<nodes_out[l].size(); i++) {
        ifft_offset[i] = nodes_out[l][i]->children[0]->idx * nsurf;
      }
      // calculate interaction_offset_f & interaction_count_offset
      std::vector<size_t> interaction_offset_f;
      std::vector<size_t> interaction_count_offset;
      for(size_t i=0; i<nodes_in.size(); i++) {
        nodes_in[i]->idx_M2L = i;  // node_id: node's index in nodes_in list
      }
      size_t n_blk1 = nodes_out[l].size() * sizeof(real_t) / CACHE_SIZE;
      if(n_blk1==0) n_blk1 = 1;
      size_t interaction_count_offset_ = 0;
      size_t fftsize = 2 * 8 * n3_;
      for(size_t blk1=0; blk1<n_blk1; blk1++) {
        size_t blk1_start=(nodes_out[l].size()* blk1   )/n_blk1;
        size_t blk1_end  =(nodes_out[l].size()*(blk1+1))/n_blk1;
        for(size_t k=0; k<mat_cnt; k++) {
          for(size_t i=blk1_start; i<blk1_end; i++) {
            NodePtrs_t& M2L_list = nodes_out[l][i]->M2L_list;
            if(M2L_list[k]) {
              interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fftsize);   // node_in's displacement in fft_in
              interaction_offset_f.push_back(        i           * fftsize);   // node_out's displacement in fft_out
              interaction_count_offset_++;
            }
          }
          interaction_count_offset.push_back(interaction_count_offset_);
        }
      }
      m2ldata[l].fft_offset     = fft_offset;
      m2ldata[l].ifft_offset    = ifft_offset;
      m2ldata[l].interaction_offset_f = interaction_offset_f;
      m2ldata[l].interaction_count_offset = interaction_count_offset;
    }
  }

  void ModifiedHelmholtzFMM::hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                       AlignedVec& fft_in, AlignedVec& fft_out, std::vector<AlignedVec>& matrix_M2L) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fftsize = 2 * NCHILD * n3_;
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

  void ModifiedHelmholtzFMM::fft_up_equiv(std::vector<size_t>& fft_offset, RealVec& all_up_equiv, AlignedVec& fft_in) {
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

    size_t fftsize = 2 * NCHILD * n3_;
    AlignedVec fftw_in(n3 * NCHILD);
    AlignedVec fftw_out(fftsize);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                          (real_t*)&fftw_in[0], nullptr, 1, n3,
                                          (fft_complex*)(&fftw_out[0]), nullptr, 1, n3_, 
                                          FFTW_ESTIMATE);

    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fftsize, 0);
      RealVec equiv_t(NCHILD*n3, 0.);

      real_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's up_equiv in all_up_equiv, size=8*nsurf
      real_t* up_equiv_f = &fft_in[fftsize*node_idx];   // offset ptr of node_idx in fft_in vector, size=fftsize

      for(int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          equiv_t[idx+j0*n3] = up_equiv[j0*nsurf+k];
      }
      fft_execute_dft_r2c(plan, &equiv_t[0], (fft_complex*)&buffer[0]);
      for(int j=0; j<n3_; j++) {
        for(int k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(n3_*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(n3_*k+j)+1];
        }
      }
    }
    fft_destroy_plan(plan);
  }

  void ModifiedHelmholtzFMM::ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out, RealVec& all_dn_equiv) {
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

    size_t fftsize = 2 * NCHILD * n3_;
    AlignedVec fftw_in(fftsize);
    AlignedVec fftw_out(n3 * NCHILD);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                    (fft_complex*)(&fftw_in[0]), nullptr, 1, n3_, 
                    (real_t*)(&fftw_out[0]), nullptr, 1, n3, 
                    FFTW_ESTIMATE);

    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fftsize, 0);
      RealVec buffer1(fftsize, 0);
      real_t* dn_check_f = &fft_out[fftsize*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      real_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * nsurf
      for(int j=0; j<n3_; j++)
        for(int k=0; k<NCHILD; k++) {
          buffer0[2*(n3_*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(n3_*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(plan, (fft_complex*)&buffer0[0], (real_t*)(&buffer1[0]));
      for(int k=0; k<nsurf; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dn_equiv[nsurf*j0+k] += buffer1[idx+j0*n3];
      }
    }
    fft_destroy_plan(plan);
  }

  void ModifiedHelmholtzFMM::M2L(Nodes_t& nodes) {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    int fft_size = 2 * 8 * n3_;
    int num_nodes = nodes.size();
    int num_coords = REL_COORD[M2L_Type].size();   // number of relative coords for M2L_Type

    // allocate memory
    RealVec all_up_equiv, all_dn_equiv;
    all_up_equiv.reserve(num_nodes*nsurf);
    all_dn_equiv.reserve(num_nodes*nsurf);
    std::vector<AlignedVec> matrix_M2L(num_coords, AlignedVec(fft_size*NCHILD, 0));

    // setup ifstream of M2L precomputation matrix
    std::string fname = "modified_helmholtz";   // precomputation matrix file name
    fname += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(p) + "_" + "l" + std::to_string(depth);
    fname += ".dat";
    std::ifstream ifile(fname, std::ifstream::binary);
    ifile.seekg(0, ifile.end);
    size_t fsize = ifile.tellg();   // file size in bytes
    size_t msize = NCHILD * NCHILD * n3_ * 2 * sizeof(real_t);   // size in bytes for each M2L matrix
    ifile.seekg(fsize - depth*num_coords*msize, ifile.beg);   // go to the start of M2L section
    
    // collect all upward equivalent charges
    #pragma omp parallel for collapse(2)
    for(int i=0; i<num_nodes; ++i) {
      for(int j=0; j<nsurf; ++j) {
        all_up_equiv[i*nsurf+j] = nodes[i].up_equiv[j];
        all_dn_equiv[i*nsurf+j] = nodes[i].dn_equiv[j];
      }
    }
    // FFT-accelerate M2L
    for(int l=0; l<depth; ++l) {
      // load M2L matrix for current level
      for(int i=0; i<num_coords; ++i) {
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
    for(int i=0; i<num_nodes; ++i) {
      for(int j=0; j<nsurf; ++j) {
        // nodes[i].up_equiv[j] = all_up_equiv[i*nsurf+j];
        nodes[i].dn_equiv[j] = all_dn_equiv[i*nsurf+j];
      }
    }
    ifile.close();   // close ifstream
  }
}  // end namespace exafmm_t
