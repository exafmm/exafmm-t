#include "laplace.h"

namespace exafmm_t {
  int P;
  int NSURF;
  int MAXLEVEL;
  M2LData M2Ldata;

  //! blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! blas gemv with row major data
  void gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
    char trans = 'T';
    real_t alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
#if FLOAT
    sgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    dgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    RealVec WORK(LWORK);
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  RealVec transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEF = 1.0/(16*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = sx - tx;
        simdvec sy(src_coord[3*s+1]);
        sy = sy - ty;
        simdvec sz(src_coord[3*s+2]);
        sz = sz - tz;
        simdvec sv(src_value[s]);
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;
        tv += invR * sv;
      }
      tv *= coef;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[t+k] += tv[k];
      }
    }
    //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*20);
  }

  void gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEFP = 1.0/(16*4*M_PI);   // factor 16 comes from the simd rsqrt function
    const real_t COEFG = -1.0/(16*16*16*4*M_PI);
    simdvec coefp(COEFP);
    simdvec coefg(COEFG);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv0(zero);
      simdvec tv1(zero);
      simdvec tv2(zero);
      simdvec tv3(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = tx - sx;
        simdvec sy(src_coord[3*s+1]);
        sy = ty - sy;
        simdvec sz(src_coord[3*s+2]);
        sz = tz - sz;
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;
        simdvec invR3 = (invR*invR) * invR;
        simdvec sv(src_value[s]);
        tv0 += sv*invR;
        sv *= invR3;
        tv1 += sv*sx;
        tv2 += sv*sy;
        tv3 += sv*sz;
      }
      tv0 *= coefp;
      tv1 *= coefg;
      tv2 *= coefg;
      tv3 *= coefg;
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
        trg_value[0+4*(t+k)] += tv0[k];
        trg_value[1+4*(t+k)] += tv1[k];
        trg_value[2+4*(t+k)] += tv2[k];
        trg_value[3+4*(t+k)] += tv3[k];
      }
    }
    //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*27);
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1.);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0.);
      potential_P2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }

  void P2M(NodePtrs& leafs) {
    real_t c[3] = {0.0};
    std::vector<RealVec> up_check_surf;
    up_check_surf.resize(MAXLEVEL+1);
    for(size_t level = 0; level <= MAXLEVEL; level++) {
      up_check_surf[level].resize(NSURF*3);
      up_check_surf[level] = surface(P,c,2.95,level);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->level;
      real_t scal = pow(0.5, level);  // scaling factor of UC2UE precomputation matrix
      // calculate upward check potential induced by sources' charges
      RealVec checkCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        checkCoord[3*k+0] = up_check_surf[level][3*k+0] + leaf->xmin[0];
        checkCoord[3*k+1] = up_check_surf[level][3*k+1] + leaf->xmin[1];
        checkCoord[3*k+2] = up_check_surf[level][3*k+2] + leaf->xmin[2];
      }
      potential_P2P(leaf->src_coord, leaf->src_value, checkCoord, leaf->up_equiv);
      // convert upward check potential to upward equivalent charge
      RealVec buffer(NSURF);
      RealVec equiv(NSURF);
      gemv(NSURF, NSURF, &matrix_UC2E_U[0], &(leaf->up_equiv[0]), &buffer[0]);
      gemv(NSURF, NSURF, &matrix_UC2E_V[0], &buffer[0], &equiv[0]);
      // scale the check-to-equivalent conversion (precomputation)
      for(int k=0; k<NSURF; k++)
        leaf->up_equiv[k] = scal * equiv[k];
    }
  }

  void M2M(Node* node) {
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
        Node* child = node->children[octant];
        RealVec buffer(NSURF);
        gemv(NSURF, NSURF, &(matrix_M2M[octant][0]), &child->up_equiv[0], &buffer[0]);
        for(int k=0; k<NSURF; k++) {
          node->up_equiv[k] += buffer[k];
        }
      }
    }
  }

  void L2L(Node* node) {
    if(node->is_leaf) return;
    // evaluate child's downward check potential from parent's downward check potential
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node* child = node->children[octant];
        RealVec buffer(NSURF);
        gemv(NSURF, NSURF, &(matrix_L2L[octant][0]), &node->dn_equiv[0], &buffer[0]);
        for(int k=0; k<NSURF; k++)
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

  void L2P(NodePtrs& leafs) {
    real_t c[3] = {0.0};
    std::vector<RealVec> dn_equiv_surf;
    dn_equiv_surf.resize(MAXLEVEL+1);
    for(size_t level = 0; level <= MAXLEVEL; level++) {
      dn_equiv_surf[level].resize(NSURF*3);
      dn_equiv_surf[level] = surface(P,c,2.95,level);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->level;
      real_t scal = pow(0.5, level);
      // convert downward check potential to downward equivalent charge
      RealVec buffer(NSURF);
      RealVec equiv(NSURF);
      gemv(NSURF, NSURF, &matrix_DC2E_U[0], &(leaf->dn_equiv[0]), &buffer[0]);
      gemv(NSURF, NSURF, &matrix_DC2E_V[0], &buffer[0], &equiv[0]);
      // scale the check-to-equivalent conversion (precomputation)
      for(int k=0; k<NSURF; k++)
        leaf->dn_equiv[k] = scal * equiv[k];
      // calculate targets' potential & gradient induced by downward equivalent charge
      RealVec equivCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        equivCoord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->xmin[0];
        equivCoord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->xmin[1];
        equivCoord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->xmin[2];
      }
      gradient_P2P(equivCoord, leaf->dn_equiv, leaf->trg_coord, leaf->trg_value);
    }
  }

  void P2L(Nodes& nodes) {
    Nodes& targets = nodes;
    real_t c[3] = {0.0};
    std::vector<RealVec> dn_check_surf;
    dn_check_surf.resize(MAXLEVEL+1);
    for(size_t level = 0; level <= MAXLEVEL; level++) {
      dn_check_surf[level].resize(NSURF*3);
      dn_check_surf[level] = surface(P,c,1.05,level);
    }
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = &targets[i];
      NodePtrs& sources = target->P2L_list;
      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
        RealVec trg_check_coord(NSURF*3);
        int level = target->level;
        // target node's check coord = relative check coord + node's origin
        for(int k=0; k<NSURF; k++) {
          trg_check_coord[3*k+0] = dn_check_surf[level][3*k+0] + target->xmin[0];
          trg_check_coord[3*k+1] = dn_check_surf[level][3*k+1] + target->xmin[1];
          trg_check_coord[3*k+2] = dn_check_surf[level][3*k+2] + target->xmin[2];
        }
        potential_P2P(source->src_coord, source->src_value, trg_check_coord, target->dn_equiv);
      }
    }
  }

  void M2P(NodePtrs& leafs) {
    NodePtrs& targets = leafs;
    real_t c[3] = {0.0};
    std::vector<RealVec> up_equiv_surf;
    up_equiv_surf.resize(MAXLEVEL+1);
    for(size_t level = 0; level <= MAXLEVEL; level++) {
      up_equiv_surf[level].resize(NSURF*3);
      up_equiv_surf[level] = surface(P,c,1.05,level);
    }
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      NodePtrs& sources = target->M2P_list;
      for(int j=0; j<sources.size(); j++) {
      Node* source = sources[j];
        RealVec sourceEquivCoord(NSURF*3);
        int level = source->level;
        // source node's equiv coord = relative equiv coord + node's origin
        for(int k=0; k<NSURF; k++) {
          sourceEquivCoord[3*k+0] = up_equiv_surf[level][3*k+0] + source->xmin[0];
          sourceEquivCoord[3*k+1] = up_equiv_surf[level][3*k+1] + source->xmin[1];
          sourceEquivCoord[3*k+2] = up_equiv_surf[level][3*k+2] + source->xmin[2];
        }
        gradient_P2P(sourceEquivCoord, source->up_equiv, target->trg_coord, target->trg_value);
      }
    }
  }

  void P2P(NodePtrs& leafs) {
    NodePtrs& targets = leafs;   // assume sources == targets
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      NodePtrs& sources = target->P2P_list;
      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
        gradient_P2P(source->src_coord, source->src_value, target->trg_coord, target->trg_value);
      }
    }
  }

  void M2L_setup(NodePtrs& nonleafs) {
    int n1 = P * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t mat_cnt = REL_COORD[M2L_Type].size();
    // construct nodes_out & nodes_in
    NodePtrs& nodes_out = nonleafs;
    std::set<Node*> nodes_in_;
    for(size_t i=0; i<nodes_out.size(); i++) {
      NodePtrs& M2L_list = nodes_out[i]->M2L_list;
      for(size_t k=0; k<mat_cnt; k++) {
        if(M2L_list[k])
          nodes_in_.insert(M2L_list[k]);
      }
    }
    NodePtrs nodes_in;
    for(std::set<Node*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
      nodes_in.push_back(*node);
    }
    // prepare fft displ & fft scal
    std::vector<size_t> fft_offset(nodes_in.size());
    std::vector<size_t> ifft_offset(nodes_out.size());
    RealVec ifft_scale(nodes_out.size());
    for(size_t i=0; i<nodes_in.size(); i++) {
      fft_offset[i] = nodes_in[i]->children[0]->idx * NSURF;
    }
    for(size_t i=0; i<nodes_out.size(); i++) {
      int level = nodes_out[i]->level+1;
      ifft_offset[i] = nodes_out[i]->children[0]->idx * NSURF;
      ifft_scale[i] = powf(2.0, level);
    }
    // calculate interaction_offset_f & interaction_count_offset
    std::vector<size_t> interaction_offset_f;
    std::vector<size_t> interaction_count_offset;
    for(size_t i=0; i<nodes_in.size(); i++) {
     nodes_in[i]->idx_M2L=i;
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
          NodePtrs& M2L_list = nodes_out[i]->M2L_list;
          if(M2L_list[k]) {
            interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fftsize);   // node_in dspl
            interaction_offset_f.push_back(        i           * fftsize);   // node_out dspl
            interaction_count_offset_++;
          }
        }
        interaction_count_offset.push_back(interaction_count_offset_);
      }
    }
    M2Ldata.fft_offset     = fft_offset;
    M2Ldata.ifft_offset    = ifft_offset;
    M2Ldata.ifft_scale    = ifft_scale;
    M2Ldata.interaction_offset_f = interaction_offset_f;
    M2Ldata.interaction_count_offset = interaction_count_offset;
  }

  void hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                       AlignedVec& fft_in, AlignedVec& fft_out) {
    int n1 = P * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fftsize = 2 * 8 * n3_;
    AlignedVec zero_vec0(fftsize, 0.);
    AlignedVec zero_vec1(fftsize, 0.);

    size_t mat_cnt = matrix_M2L.size();
    size_t blk1_cnt = interaction_count_offset.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
    std::vector<real_t*> IN_(BLOCK_SIZE*blk1_cnt*mat_cnt);
    std::vector<real_t*> OUT_(BLOCK_SIZE*blk1_cnt*mat_cnt);

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
      for(size_t k=0; k<n3_; k++) {
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
    //Profile::Add_FLOP(8*8*8*(interaction_offset_f.size()/2)*n3_);
  }

  void fft_up_equiv(std::vector<size_t>& fft_offset,
                   RealVec& all_up_equiv, AlignedVec& fft_in) {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(P-2);
    RealVec surf = surface(P, c, (real_t)(P-1), 0, true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(P-1-surf[i*3]+0.5))
             + ((size_t)(P-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(P-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(n3 * NCHILD);
    AlignedVec fftw_out(fftsize);
    int dim[3] = {2*P, 2*P, 2*P};
    fft_plan m2l_list_fftplan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                (real_t*)&fftw_in[0], nullptr, 1, n3,
                                (fft_complex*)(&fftw_out[0]), nullptr, 1, n3_,
                                FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fftsize, 0);
      real_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's upward_equiv in all_up_equiv, size=8*NSURF
      // upward_equiv_fft (input of r2c) here should have a size of N3*NCHILD
      // the node_idx's chunk of fft_out has a size of 2*N3_*NCHILD
      // since it's larger than what we need,  we can use fft_out as fftw_in buffer here
      real_t* up_equiv_f = &fft_in[fftsize*node_idx]; // offset ptr of node_idx in fft_in vector, size=fftsize
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<(int)NCHILD; j0++)
          // up_equiv_f[idx+j0*n3] = up_equiv[j0*NSURF+k] * fft_scal[node_idx];
          up_equiv_f[idx+j0*n3] = up_equiv[j0*NSURF+k];
      }
      fft_execute_dft_r2c(m2l_list_fftplan, up_equiv_f, (fft_complex*)&buffer[0]);
      for(size_t j=0; j<n3_; j++) {
        for(size_t k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(n3_*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(n3_*k+j)+1];
        }
      }
    }
    fft_destroy_plan(m2l_list_fftplan);
  }

  void ifft_dn_check(std::vector<size_t>& ifft_offset, RealVec& ifft_scal,
                       AlignedVec& fft_out, RealVec& all_dn_equiv) {
    int n1 = P * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(P-2);
    RealVec surf = surface(P, c, (real_t)(P-1), 0, true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(P*2-0.5-surf[i*3]))
             + ((size_t)(P*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(P*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(fftsize);
    AlignedVec fftw_out(n3 * NCHILD);
    int dim[3] = {2*P, 2*P, 2*P};
    fft_plan m2l_list_ifftplan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                                 (fft_complex*)&fftw_in[0], nullptr, 1, n3_,
                                 (real_t*)(&fftw_out[0]), nullptr, 1, n3,
                                 FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fftsize, 0);
      RealVec buffer1(fftsize, 0);
      real_t* dn_check_f = &fft_out[fftsize*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      real_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * NSURF
      for(size_t j=0; j<n3_; j++)
        for(size_t k=0; k<NCHILD; k++) {
          buffer0[2*(n3_*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(n3_*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dn_equiv[NSURF*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
      }
    }
    fft_destroy_plan(m2l_list_ifftplan);
  }

  void M2L(Nodes& nodes) {
    int n1 = P * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t numNodes = nodes.size();
    RealVec all_up_equiv(numNodes*NSURF);
    RealVec all_dn_equiv(numNodes*NSURF);
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        all_up_equiv[i*NSURF+j] = nodes[i].up_equiv[j];
        all_dn_equiv[i*NSURF+j] = nodes[i].dn_equiv[j];
      }
    }
    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fft_in(M2Ldata.fft_offset.size()*fftsize, 0.);
    AlignedVec fft_out(M2Ldata.ifft_offset.size()*fftsize, 0.);

    fft_up_equiv(M2Ldata.fft_offset, all_up_equiv, fft_in);
    hadamard_product(M2Ldata.interaction_count_offset, M2Ldata.interaction_offset_f, fft_in, fft_out);
    ifft_dn_check(M2Ldata.ifft_offset, M2Ldata.ifft_scale, fft_out, all_dn_equiv);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        nodes[i].up_equiv[j] = all_up_equiv[i*NSURF+j];
        nodes[i].dn_equiv[j] = all_dn_equiv[i*NSURF+j];
      }
    }
  }
}//end namespace
