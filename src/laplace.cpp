#include "laplace.h"
#include "laplace_cuda.h"
#include "profile.h"

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

  void gradientP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
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
	if (r2 != 0) {
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
      std::vector<int> sources_idx = target->P2Llist_idx;
      for(int j=0; j<sources_idx.size(); j++) {
        Node* source = &nodes[sources_idx[j]];
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

  void M2P(Nodes &nodes, std::vector<Node*>& leafs) {
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
      std::vector<int> sources_idx= target->M2Plist_idx;
      for(int j=0; j<sources_idx.size(); j++) {
      Node* source = &nodes[sources_idx[j]];
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

 void P2P(Nodes &nodes, std::vector<int> leafs_idx) {
    std::vector<real_t> nodes_coord; 
    std::vector<int>nodes_coord_idx;
    int nodes_coord_idx_cnt = 0;

    std::vector<real_t> nodes_pt_src;
    std::vector<int> nodes_pt_src_idx;
    int nodes_pt_src_idx_cnt = 0;
    
    std::vector<int>P2Plists;
    std::vector<int>P2Plists_idx;
    int P2Plists_idx_cnt = 0;
    
    Profile::Tic("memcpy vector to array", true);
    for(int i=0; i<nodes.size(); i++) {
      Node *node = &nodes[i];
     
      RealVec& pt_coord = node->pt_coord;
      nodes_coord.insert(nodes_coord.end() , pt_coord.begin(), pt_coord.end());
      nodes_coord_idx.push_back(nodes_coord_idx_cnt);
      nodes_coord_idx_cnt += pt_coord.size();

      RealVec& pt_src = node->pt_src;
      nodes_pt_src.insert(nodes_pt_src.end(), pt_src.begin(), pt_src.end());
      nodes_pt_src_idx.push_back(nodes_pt_src_idx_cnt);
      nodes_pt_src_idx_cnt += pt_src.size();
    }
    nodes_coord_idx.push_back(nodes_coord_idx_cnt);
    nodes_pt_src_idx.push_back(nodes_pt_src_idx_cnt);
    int trg_val_size = 0;
    std::vector<int> targets_idx = leafs_idx;
    for(int i=0; i<targets_idx.size(); i++) {
      Node* target = &nodes[targets_idx[i]];
      std::vector<int> sources_idx = target->P2Plist_idx;
      P2Plists.insert(P2Plists.end(), sources_idx.begin(), sources_idx.end());
      P2Plists_idx.push_back(P2Plists_idx_cnt);
      P2Plists_idx_cnt += sources_idx.size();
    
      trg_val_size += target->pt_trg.size();
    }
    std::vector<real_t> trg_val(4*nodes_coord_idx_cnt/3);
    P2Plists_idx.push_back(P2Plists_idx_cnt);
    Profile::Toc();
    
    P2PGPU(leafs_idx, nodes_coord, nodes_coord_idx, nodes_pt_src, nodes_pt_src_idx, P2Plists, P2Plists_idx, trg_val);
    int pt_trg_count = 0;
    Profile::Tic("memcpy array to vec", true);
    for(int i=0; i<targets_idx.size(); i++) {
      Node* target = &nodes[targets_idx[i]];
      for(int j=0;j<target->pt_trg.size();j++) {
        target->pt_trg[j] = trg_val[pt_trg_count++];
      }
    }
    Profile::Toc();
  }
  
  void hadamardProduct(real_t *kernel, AlignedVec& equiv, real_t *check) {
    int n3_ = (int)(equiv.size()/2);
    for(int k=0; k<n3_; ++k) {
      int real = 2*k+0;
      int imag = 2*k+1;
      check[real] += kernel[real]*equiv[real] - kernel[imag]*equiv[imag];
      check[imag] += kernel[real]*equiv[imag] + kernel[imag]*equiv[real];
    }
  }

  void FFT_UpEquiv(Nodes& nodes, std::vector<int> &M2Lsources_idx) {
    // define constants
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map.size(); i++) {
      // mapping: upward equiv surf -> conv grid
      map[i] = ((size_t)(MULTIPOLE_ORDER-1-surf[i*3]+0.5))
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+2]+0.5)) * n1 * n1;
    }
    // create dft plan for upward equiv
    AlignedVec in(n3);
    AlignedVec out(2*n3_);
    int dim[3] = {n1, n1, n1};
    fft_plan plan = fft_plan_dft_r2c(3, dim, &in[0], (fft_complex*)(&out[0]), FFTW_ESTIMATE);
    // evaluate dft of upward equivalent of sources
    #pragma omp parallel for
    for(int i=0; i<M2Lsources_idx.size(); ++i) {
      Node* source = &nodes[M2Lsources_idx[i]];
      source->up_equiv_fft.resize(2*n3_);
      // upward equiv on convolution grid
      AlignedVec upequiv(n3, 0);
      for(int j=0; j<NSURF; ++j) {
        int conv_id = map[j];
        upequiv[conv_id] = source->upward_equiv[j];
      }
      // dft of upward equiv on convolution grid
      fft_execute_dft_r2c(plan, &upequiv[0], (fft_complex*)(&(source->up_equiv_fft[0])));
    }
    fft_destroy_plan(plan);
  }
  
  void FFT_Check2Equiv(Nodes& nodes, std::vector<int> &M2Ltargets_idx, std::vector<real_t> &check) {
    // define constants
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    // calculate mapping
    std::vector<size_t> map2(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map2.size(); i++) {
      // mapping: conv grid -> downward check surf
      map2[i] = ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3]))
              + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+1])) * n1
              + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    // create idft plan for downward check
    AlignedVec in2(2*n3_);
    AlignedVec out2(n3);
    int dim[3] = {n1, n1, n1};
    fft_plan iplan = fft_plan_dft_c2r(3, dim, (fft_complex*)(&in2[0]), &out2[0], FFTW_ESTIMATE);
    #pragma omp parallel for
    for(int i=0; i<M2Ltargets_idx.size(); ++i) {
      Node* target = &nodes[M2Ltargets_idx[i]];
      AlignedVec dnCheck(n3);
      fft_execute_dft_c2r(iplan, (fft_complex*)(&check[i*2*n3_]), &dnCheck[0]);
      real_t scale = powf(2, target->depth);
      for(int j=0; j<NSURF; ++j) {
        int conv_id = map2[j];
        target->dnward_equiv[j] += dnCheck[conv_id] * scale;
      }
    }
  }

  void M2L(Nodes& nodes, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx) {
    // define constants
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    Profile::Tic("FFT_UpEquiv", true);
    FFT_UpEquiv(nodes, M2Lsources_idx);
    Profile::Toc();
    Profile::Tic("hadamard", true);
    std::vector<real_t> nodes_up_equiv_fft(nodes.size()*2*n3_);
    std::vector<int> M2Llist_start_idx;
    std::vector<int> M2Llists;
    int M2Llist_start_idx_cnt = 0;
    
    std::vector<int> M2LRelPos_start_idx;
    std::vector<int> M2LRelPoss;
    int M2LRelPos_start_idx_cnt = 0;
    for (int i = 0; i < nodes.size(); ++i)
    {
      Node node = nodes[i];
      nodes_up_equiv_fft.insert(nodes_up_equiv_fft.end(), node.up_equiv_fft.begin(), node.up_equiv_fft.end());
    }
    for(int i=0; i<M2Ltargets_idx.size(); ++i) {
      Node* target = &nodes[M2Ltargets_idx[i]];
      
      M2Llist_start_idx.push_back(M2Llist_start_idx_cnt);
      M2Llists.insert(M2Llists.end(), target->M2Llist_idx.begin(), target->M2Llist_idx.end());
      M2Llist_start_idx_cnt += target->M2Llist_idx.size();

      M2LRelPos_start_idx.push_back(M2LRelPos_start_idx_cnt);
      M2LRelPoss.insert(M2LRelPoss.end(), target->M2LRelPos.begin(),target->M2LRelPos.end());
       M2LRelPos_start_idx_cnt += target->M2LRelPos.size();
    }
    M2Llist_start_idx.push_back(M2Llist_start_idx_cnt);
    M2LRelPos_start_idx.push_back(M2LRelPos_start_idx_cnt);
    std::vector<real_t> check(2*n3_*M2Ltargets_idx.size(), 0.);
    HadmardGPU(M2Ltargets_idx, nodes_up_equiv_fft, M2Llist_start_idx, M2Llists, M2LRelPos_start_idx, M2LRelPoss, mat_M2L_Helper, check);
    Profile::Toc();
    Profile::Tic("FFT_Check2Equiv", true);
    FFT_Check2Equiv(nodes, M2Ltargets_idx, check);
    Profile::Toc();
  }
}//end namespace
