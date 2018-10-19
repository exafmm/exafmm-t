#include "laplace.h"
#include "laplace_cuda.h"

namespace exafmm_t {
  int MULTIPOLE_ORDER;
  int NSURF;
  int MAXLEVEL;
  M2LData M2Ldata;

  //! using blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
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

void P2PKernel_test(real_t* trg_coord, real_t* trg_value, real_t* src_coord, real_t* src_value, int leafs_cnt, int ncrit, int adj_cnt) {
  int src_cnt = ncrit*adj_cnt;
  const real_t COEFP = 1.0/(2*4*M_PI);  
  const real_t COEFG = -1.0/(4*2*2*6*M_PI);
  for (int i=0;i<leafs_cnt;i++) {
    for (int j = 0; j < ncrit; j++)
    {
      int idx = i * ncrit + j;
      real_t tx = trg_coord[3*idx+0];
      real_t ty = trg_coord[3*idx+1];
      real_t tz = trg_coord[3*idx+2];
      real_t tv0=0;
      real_t tv1=0;
      real_t tv2=0;
      real_t tv3=0;
      
      for(int k=i*src_cnt; k<i*src_cnt+src_cnt; k++) {
        real_t sx = src_coord[3*k+0] - tx;
        real_t sy = src_coord[3*k+1] - ty;
        real_t sz = src_coord[3*k+2] - tz;
        real_t r2 = sx*sx + sy*sy + sz*sz;
        real_t sv = src_value[k];
        if (r2 != 0)
        {
          real_t invR = 1.0/sqrt(r2);
          real_t invR3 = invR*invR*invR;
          tv0 += invR*sv;
          sv *= invR3;
          tv1 += sv*sx;
          tv2 += sv*sy;
          tv3 += sv*sz;
        }
      }
      tv0 *= COEFP;
    tv1 *= COEFG;
    tv2 *= COEFG;
    tv3 *= COEFG;
    trg_value[4*idx+0] += tv0;  
    trg_value[4*idx+1] += tv1;
    trg_value[4*idx+2] += tv2;
    trg_value[4*idx+3] += tv3;
    }
  } 
}

  void potentialP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero((real_t)0);
    const real_t COEF = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
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

void gradientP2P(RealVec& src_coord, RealVec& src_value, 
    RealVec& trg_coord, RealVec& trg_value) {
	//std::cout<<"src_coord  src_val  "<<src_coord.size()<<"  "<<src_value.size()<<std::endl;
	const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);
	int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int i=0; i<trg_cnt; i++) {
    	real_t tx = trg_coord[3*i+0];
    	real_t ty = trg_coord[3*i+1];
    	real_t tz = trg_coord[3*i+2];
    	real_t tv0=0;
    	real_t tv1=0;
    	real_t tv2=0;
    	real_t tv3=0;
    	for(int j=0; j<src_cnt; j++) {
    		real_t sx = src_coord[3*j+0] - tx;
	        real_t sy = src_coord[3*j+1] - ty;
	        real_t sz = src_coord[3*j+2] - tz;
	        real_t r2 = sx*sx + sy*sy + sz*sz;	
	        real_t sv = src_value[j];
	        if (r2 != 0)
	        {
	          	real_t invR = 1.0/sqrt(r2);
	          	real_t invR3 = invR*invR*invR;
	          	tv0 += invR*sv;
	          	sv *= invR3;
	          	tv1 += sv*sx;
        	   	tv2 += sv*sy;
        		tv3 += sv*sz;
	        }
		}
		tv0 *= COEFP;
		tv1 *= COEFG;
      	tv2 *= COEFG;
      	tv3 *= COEFG;
    	trg_value[4*i+0] += tv0;
    	trg_value[4*i+1] += tv1;
    	trg_value[4*i+2] += tv2;
    	trg_value[4*i+3] += tv3;
    }
}

//! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1.);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0.);
      potentialP2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }

  void P2M(std::vector<Node*>& leafs) {
    real_t c[3] = {0.0};
    std::vector<RealVec> upwd_check_surf;
    upwd_check_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      upwd_check_surf[depth].resize(NSURF*3);
      upwd_check_surf[depth] = surface(MULTIPOLE_ORDER,c,2.95,depth);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);    // scaling factor of UC2UE precomputation matrix source charge -> check surface potential
      RealVec checkCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        checkCoord[3*k+0] = upwd_check_surf[level][3*k+0] + leaf->coord[0];
        checkCoord[3*k+1] = upwd_check_surf[level][3*k+1] + leaf->coord[1];
        checkCoord[3*k+2] = upwd_check_surf[level][3*k+2] + leaf->coord[2];
      }
      potentialP2P(leaf->pt_coord, leaf->pt_src, checkCoord, leaf->upward_equiv);
      RealVec buffer(NSURF);
      RealVec equiv(NSURF);
      gemm(1, NSURF, NSURF, &(leaf->upward_equiv[0]), &M2M_V[0], &buffer[0]);
      gemm(1, NSURF, NSURF, &buffer[0], &M2M_U[0], &equiv[0]);
      for(int k=0; k<NSURF; k++)
        leaf->upward_equiv[k] = scal * equiv[k];
    }
  }

  void M2M(Node* node) {
    if(node->IsLeaf()) return;
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        M2M(node->child[octant]);
    }
    #pragma omp taskwait
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        Node* child = node->child[octant];
        RealVec buffer(NSURF);
        gemm(1, NSURF, NSURF, &child->upward_equiv[0], &(mat_M2M[octant][0]), &buffer[0]);
        for(int k=0; k<NSURF; k++) {
          node->upward_equiv[k] += buffer[k];
        }
      }
    }
  }

  void L2L(Node* node) {
    if(node->IsLeaf()) return;
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        Node* child = node->child[octant];
        RealVec buffer(NSURF);
        gemm(1, NSURF, NSURF, &node->dnward_equiv[0], &(mat_L2L[octant][0]), &buffer[0]);
        for(int k=0; k<NSURF; k++)
          child->dnward_equiv[k] += buffer[k];
      }
    }
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        L2L(node->child[octant]);
    }
    #pragma omp taskwait
  }

  void L2P(std::vector<Node*>& leafs) {
    real_t c[3] = {0.0};
    std::vector<RealVec> dnwd_equiv_surf;
    dnwd_equiv_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      dnwd_equiv_surf[depth].resize(NSURF*3);
      dnwd_equiv_surf[depth] = surface(MULTIPOLE_ORDER,c,2.95,depth);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);
      // check surface potential -> equivalent surface charge
      RealVec buffer(NSURF);
      RealVec equiv(NSURF);
      gemm(1, NSURF, NSURF, &(leaf->dnward_equiv[0]), &L2L_V[0], &buffer[0]);
      gemm(1, NSURF, NSURF, &buffer[0], &L2L_U[0], &equiv[0]);
      for(int k=0; k<NSURF; k++)
        leaf->dnward_equiv[k] = scal * equiv[k];
      // equivalent surface charge -> target potential
      RealVec equivCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        equivCoord[3*k+0] = dnwd_equiv_surf[level][3*k+0] + leaf->coord[0];
        equivCoord[3*k+1] = dnwd_equiv_surf[level][3*k+1] + leaf->coord[1];
        equivCoord[3*k+2] = dnwd_equiv_surf[level][3*k+2] + leaf->coord[2];
      }
      gradientP2P(equivCoord, leaf->dnward_equiv, leaf->pt_coord, leaf->pt_trg);
    }
  }

  void P2L(Nodes& nodes) {
    Nodes& targets = nodes;
    real_t c[3] = {0.0};
    std::vector<RealVec> dnwd_check_surf;
    dnwd_check_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      dnwd_check_surf[depth].resize(NSURF*3);
      dnwd_check_surf[depth] = surface(MULTIPOLE_ORDER,c,1.05,depth);
    }
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = &targets[i];
      std::vector<Node*>& sources = target->P2Llist;
      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
        RealVec targetCheckCoord(NSURF*3);
        int level = target->depth;
        // target node's check coord = relative check coord + node's origin
        for(int k=0; k<NSURF; k++) {
          targetCheckCoord[3*k+0] = dnwd_check_surf[level][3*k+0] + target->coord[0];
          targetCheckCoord[3*k+1] = dnwd_check_surf[level][3*k+1] + target->coord[1];
          targetCheckCoord[3*k+2] = dnwd_check_surf[level][3*k+2] + target->coord[2];
        }
        potentialP2P(source->pt_coord, source->pt_src, targetCheckCoord, target->dnward_equiv);
      }
    }
  }

  void M2P(std::vector<Node*>& leafs) {
    std::vector<Node*>& targets = leafs;
    real_t c[3] = {0.0};
    std::vector<RealVec> upwd_equiv_surf;
    upwd_equiv_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      upwd_equiv_surf[depth].resize(NSURF*3);
      upwd_equiv_surf[depth] = surface(MULTIPOLE_ORDER,c,1.05,depth);
    }
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      std::vector<Node*>& sources = target->M2Plist;
      for(int j=0; j<sources.size(); j++) {
      Node* source = sources[j];
        RealVec sourceEquivCoord(NSURF*3);
        int level = source->depth;
        // source node's equiv coord = relative equiv coord + node's origin
        for(int k=0; k<NSURF; k++) {
          sourceEquivCoord[3*k+0] = upwd_equiv_surf[level][3*k+0] + source->coord[0];
          sourceEquivCoord[3*k+1] = upwd_equiv_surf[level][3*k+1] + source->coord[1];
          sourceEquivCoord[3*k+2] = upwd_equiv_surf[level][3*k+2] + source->coord[2];
        }
        gradientP2P(sourceEquivCoord, source->upward_equiv, target->pt_coord, target->pt_trg);
      }
    }
  }

  void P2P(std::vector<Node*>& leafs) {
    std::vector<Node*>& targets = leafs;   // assume sources == targets
    int ncrit=64;
    int coord_count = ncrit*3;
    int src_box_count = 27;
    real_t *trg_pt_coord=new real_t[targets.size()*coord_count]();
    real_t *trg_pt_trg=new real_t[targets.size()*ncrit*4]();
    real_t *src_pt_coord = new real_t[targets.size()*src_box_count*coord_count]();
    real_t *src_pt_src = new real_t[targets.size()*src_box_count*ncrit]();
    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      std::vector<Node*>& sources = target->P2Plist;
      RealVec& trg_coord = target->pt_coord;
      RealVec& trg_val = target->pt_trg;

      for (int k = 0; k < trg_coord.size(); k++)
        trg_pt_coord[i*ncrit*3+k] = trg_coord[k];
      for (int k = 0; k < trg_val.size(); k++)
        trg_pt_trg[i*ncrit*4+k] = trg_val[k];

      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
	//gradientP2P(source->pt_coord, source->pt_src, target->pt_coord, target->pt_trg);
	RealVec& src_coord = source->pt_coord; 
	RealVec& src_val = source->pt_src;

	for (int k = 0; k < src_coord.size(); k++)
           src_pt_coord[i*src_box_count*ncrit*3+(j*ncrit*3)+k] = src_coord[k];
      for (int k = 0; k < src_val.size(); k++) 
          src_pt_src[i*src_box_count*ncrit+(j*ncrit)+k] = src_val[k];
      }
   }
    P2PGPU(trg_pt_coord, trg_pt_trg, src_pt_coord, src_pt_src, targets.size(), ncrit, src_box_count);
//	P2PKernel_test(trg_pt_coord, trg_pt_trg, src_pt_coord, src_pt_src, targets.size(), ncrit, src_box_count);

    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      RealVec& trg_val = target->pt_trg;
     for(int j=0;j<trg_val.size();j++) {
        trg_val[j] = trg_pt_trg[i*ncrit*4+j];
     }
    }

    delete[] trg_pt_coord;
    delete[] src_pt_coord;
  }

  void M2LSetup(std::vector<Node*>& nonleafs) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t mat_cnt = rel_coord[M2L_Type].size();
    // construct nodes_out & nodes_in
    std::vector<Node*>& nodes_out = nonleafs;
    std::set<Node*> nodes_in_;
    for(size_t i=0; i<nodes_out.size(); i++) {
      std::vector<Node*>& M2Llist = nodes_out[i]->M2Llist;
      for(size_t k=0; k<mat_cnt; k++) {
        if(M2Llist[k]!=NULL)
          nodes_in_.insert(M2Llist[k]);
      }
    }
    std::vector<Node*> nodes_in;
    for(std::set<Node*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
      nodes_in.push_back(*node);
    }
    // prepare fft displ & fft scal
    std::vector<size_t> fft_vec(nodes_in.size());
    std::vector<size_t> ifft_vec(nodes_out.size());
    RealVec fft_scl(nodes_in.size());
    RealVec ifft_scl(nodes_out.size());
    for(size_t i=0; i<nodes_in.size(); i++) {
      fft_vec[i] = nodes_in[i]->child[0]->idx * NSURF;
      fft_scl[i] = 1;
    }
    for(size_t i=0; i<nodes_out.size(); i++) {
      int depth = nodes_out[i]->depth+1;
      ifft_vec[i] = nodes_out[i]->child[0]->idx * NSURF;
      ifft_scl[i] = powf(2.0, depth);
    }
    // calculate interac_vec & interac_dsp
    std::vector<size_t> interac_vec;
    std::vector<size_t> interac_dsp;
    for(size_t i=0; i<nodes_in.size(); i++) {
     nodes_in[i]->node_id=i;
    }
    size_t n_blk1 = nodes_out.size() * sizeof(real_t) / CACHE_SIZE;
    if(n_blk1==0) n_blk1 = 1;
    size_t interac_dsp_ = 0;
    size_t fftsize = 2 * 8 * n3_;
    for(size_t blk1=0; blk1<n_blk1; blk1++) {
      size_t blk1_start=(nodes_out.size()* blk1   )/n_blk1;
      size_t blk1_end  =(nodes_out.size()*(blk1+1))/n_blk1;
      for(size_t k=0; k<mat_cnt; k++) {
        for(size_t i=blk1_start; i<blk1_end; i++) {
          std::vector<Node*>& M2Llist = nodes_out[i]->M2Llist;
          if(M2Llist[k]!=NULL) {
            interac_vec.push_back(M2Llist[k]->node_id * fftsize);   // node_in dspl
            interac_vec.push_back(        i           * fftsize);   // node_out dspl
            interac_dsp_++;
          }
        }
        interac_dsp.push_back(interac_dsp_);
      }
    }
    M2Ldata.fft_vec     = fft_vec;
    M2Ldata.ifft_vec    = ifft_vec;
    M2Ldata.fft_scl     = fft_scl;
    M2Ldata.ifft_scl    = ifft_scl;
    M2Ldata.interac_vec = interac_vec;
    M2Ldata.interac_dsp = interac_dsp;
  }

  void M2LListHadamard(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fftsize = 2 * 8 * n3_;
    AlignedVec zero_vec0(fftsize, 0.);
    AlignedVec zero_vec1(fftsize, 0.);

    size_t mat_cnt = mat_M2L.size();
    size_t blk1_cnt = interac_dsp.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 2 / sizeof(real_t);
    std::vector<real_t*> IN_(BLOCK_SIZE*blk1_cnt*mat_cnt);
    std::vector<real_t*> OUT_(BLOCK_SIZE*blk1_cnt*mat_cnt);

    #pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++) {
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0; j<interac_cnt; j++) {
        IN_ [BLOCK_SIZE*interac_blk1 +j] = &fft_in[interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[BLOCK_SIZE*interac_blk1 +j] = &fft_out[interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec0[0];
      OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt] = &zero_vec1[0];
    }

    for(size_t blk1=0; blk1<blk1_cnt; blk1++) {
    #pragma omp parallel for
      for(size_t k=0; k<n3_; k++) {
        for(size_t mat_indx=0; mat_indx< mat_cnt; mat_indx++) {
          size_t interac_blk1 = blk1*mat_cnt+mat_indx;
          size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
          size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
          size_t interac_cnt  = interac_dsp1-interac_dsp0;
          real_t** IN = &IN_[BLOCK_SIZE*interac_blk1];
          real_t** OUT= &OUT_[BLOCK_SIZE*interac_blk1];
          real_t* M = &mat_M2L[mat_indx][k*2*NCHILD*NCHILD]; // k-th freq's (row) offset in mat_M2L[mat_indx]
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
    //Profile::Add_FLOP(8*8*8*(interac_vec.size()/2)*n3_);
  }

  void FFT_UpEquiv(std::vector<size_t>& fft_vec, RealVec& fft_scal,
                   RealVec& input_data, AlignedVec& fft_in) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER-1-surf[i*3]+0.5))
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(n3 * NCHILD);
    AlignedVec fftw_out(fftsize);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan m2l_list_fftplan = fft_plan_many_dft_r2c(3, dim, NCHILD,
                                (real_t*)&fftw_in[0], NULL, 1, n3,
                                (fft_complex*)(&fftw_out[0]), NULL, 1, n3_,
                                FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_vec.size(); node_idx++) {
      RealVec buffer(fftsize, 0);
      real_t* upward_equiv = &input_data[fft_vec[node_idx]];  // offset ptr of node's 8 child's upward_equiv in allUpwardEquiv, size=8*NSURF
      // upward_equiv_fft (input of r2c) here should have a size of N3*NCHILD
      // the node_idx's chunk of fft_out has a size of 2*N3_*NCHILD
      // since it's larger than what we need,  we can use fft_out as fftw_in buffer here
      real_t* upward_equiv_fft = &fft_in[fftsize*node_idx]; // offset ptr of node_idx in fft_in vector, size=fftsize
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<(int)NCHILD; j0++)
          upward_equiv_fft[idx+j0*n3] = upward_equiv[j0*NSURF+k] * fft_scal[node_idx];
      }
      fft_execute_dft_r2c(m2l_list_fftplan, upward_equiv_fft, (fft_complex*)&buffer[0]);
      for(size_t j=0; j<n3_; j++) {
        for(size_t k=0; k<NCHILD; k++) {
          upward_equiv_fft[2*(NCHILD*j+k)+0] = buffer[2*(n3_*k+j)+0];
          upward_equiv_fft[2*(NCHILD*j+k)+1] = buffer[2*(n3_*k+j)+1];
        }
      }
    }
    fft_destroy_plan(m2l_list_fftplan);
  }

  void FFT_Check2Equiv(std::vector<size_t>& ifft_vec, RealVec& ifft_scal,
                       AlignedVec& fft_out, RealVec& output_data) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3]))
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fftw_in(fftsize);
    AlignedVec fftw_out(n3 * NCHILD);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan m2l_list_ifftplan = fft_plan_many_dft_c2r(3, dim, NCHILD,
                                 (fft_complex*)&fftw_in[0], NULL, 1, n3_,
                                 (real_t*)(&fftw_out[0]), NULL, 1, n3,
                                 FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_vec.size(); node_idx++) {
      RealVec buffer0(fftsize, 0);
      RealVec buffer1(fftsize, 0);
      real_t* dnward_check_fft = &fft_out[fftsize*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      real_t* dnward_equiv = &output_data[ifft_vec[node_idx]];  // offset ptr for node_idx's child's dnward_equiv in allDnwardEquiv, size=numChilds * NSURF
      for(size_t j=0; j<n3_; j++)
        for(size_t k=0; k<NCHILD; k++) {
          buffer0[2*(n3_*k+j)+0] = dnward_check_fft[2*(NCHILD*j+k)+0];
          buffer0[2*(n3_*k+j)+1] = dnward_check_fft[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dnward_equiv[NSURF*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
      }
    }
    fft_destroy_plan(m2l_list_ifftplan);
  }

  void M2L(M2LData& M2Ldata, Nodes& nodes) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t numNodes = nodes.size();
    RealVec allUpwardEquiv(numNodes*NSURF);
    RealVec allDnwardEquiv(numNodes*NSURF);
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        allUpwardEquiv[i*NSURF+j] = nodes[i].upward_equiv[j];
        allDnwardEquiv[i*NSURF+j] = nodes[i].dnward_equiv[j];
      }
    }
    size_t fftsize = 2 * 8 * n3_;
    AlignedVec fft_in(M2Ldata.fft_vec.size()*fftsize, 0.);
    AlignedVec fft_out(M2Ldata.ifft_vec.size()*fftsize, 0.);

    FFT_UpEquiv(M2Ldata.fft_vec, M2Ldata.fft_scl, allUpwardEquiv, fft_in);
    M2LListHadamard(M2Ldata.interac_dsp, M2Ldata.interac_vec, fft_in, fft_out);
    FFT_Check2Equiv(M2Ldata.ifft_vec, M2Ldata.ifft_scl, fft_out, allDnwardEquiv);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        nodes[i].upward_equiv[j] = allUpwardEquiv[i*NSURF+j];
        nodes[i].dnward_equiv[j] = allDnwardEquiv[i*NSURF+j];
      }
    }
  }
}//end namespace
