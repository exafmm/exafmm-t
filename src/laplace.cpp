#include "laplace.h"
#include "laplace_cuda.h"
#include "profile.h"

namespace exafmm_t {
  int MULTIPOLE_ORDER;
  int NSURF;
  int MAXLEVEL;
  M2LData M2Ldata;

  //! using blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C, real_t beta) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0;
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

  void potentialP2P(real_t *src_coord, int src_coord_size, real_t *src_value, real_t *trg_coord, int trg_coord_size, real_t *trg_value) {
    const real_t COEF = 1.0/(2*4*M_PI);
    int src_cnt = src_coord_size / 3;
    int trg_cnt = trg_coord_size / 3;
    for(int t=0; t<trg_cnt; t++) {
      real_t tx = trg_coord[3*t+0];
      real_t ty = trg_coord[3*t+1];
      real_t tz = trg_coord[3*t+2];
      real_t tv = 0;
      for(int s=0; s<src_cnt; s++) {
        real_t sx = src_coord[3*s+0]-tx;
        real_t sy = src_coord[3*s+1]-ty;
        real_t sz = src_coord[3*s+2]-tz;
        real_t sv = src_value[s];
        real_t r2 = sx*sx + sy*sy + sz*sz;;
        if (r2 != 0) {
          real_t invR = 1.0/sqrt(r2);
          tv += invR * sv;
        }
      }
      tv *= COEF;
      trg_value[t] += tv;
    }
  }

  void gradientP2P(real_t *src_coord, int src_coord_size, real_t *src_value, real_t *trg_coord, int trg_coord_size, real_t *trg_value) {
    const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);
    int src_cnt = src_coord_size / 3;
    int trg_cnt = trg_coord_size / 3;
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
      potentialP2P(&src_coord[0], src_coord.size(), &src_value[0], &trg_coord[0], trg_coord.size(), &trg_value[0]);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }
  
  RealVec surface_test(int p, real_t alpha){
    size_t n_=(6*(p-1)*(p-1)+2);
    RealVec coord(n_*3);
    coord[0]=coord[1]=coord[2]=-1.0;
    size_t cnt=1;
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=-1.0;
        coord[cnt*3+1]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+2]=(2.0*j-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*i-p+1)/(p-1);
        coord[cnt*3+1]=-1.0;
        coord[cnt*3+2]=(2.0*(j+1)-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+1]=(2.0*j-p+1)/(p-1);
        coord[cnt*3+2]=-1.0;
        cnt++;
      }
    for(size_t i=0;i<(n_/2)*3;i++) coord[cnt*3+i]=-coord[i];
    for(size_t i=0;i<n_;i++){
      coord[i*3+0]= ((coord[i*3+0]+1)*alpha-alpha+1);
      coord[i*3+1]= ((coord[i*3+1]+1)*alpha-alpha+1);
      coord[i*3+2]= ((coord[i*3+2]+1)*alpha-alpha+1);
    }
    return coord;
  }
  
  void P2M(Nodes &nodes, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, int ncrit, RealVec &upward_equiv) {
    RealVec checkCoord = surface_test(MULTIPOLE_ORDER,2.95);
    RealVec r(leafs_idx.size());
    RealVec leaf_xyz(3*leafs_idx.size());
    #pragma omp parallel for
    for(int i=0; i<leafs_idx.size(); i++) {
      Node* leaf = &nodes[leafs_idx[i]];
      int level = leaf->depth;
      real_t scal = powf(0.5, level);    // scaling factor of UC2UE precomputation matrix source charge -> check surface potential
      r[i] = 0.5*scal;
      leaf_xyz[3*i+0] = leaf->coord[0];
      leaf_xyz[3*i+1] = leaf->coord[1];
      leaf_xyz[3*i+2] = leaf->coord[2];
    }
    P2MGPU(leafs_idx, nodes_coord, nodes_pt_src, nodes_pt_src_idx, checkCoord, checkCoord.size(), upward_equiv, r, leaf_xyz, ncrit);
    #pragma omp parallel for
    for(int i=0; i<leafs_idx.size(); i++) {
      Node* leaf = &nodes[leafs_idx[i]];
      int level = leaf->depth;
      real_t scal = powf(0.5, level);
      
      for(int k=0; k<NSURF; k++) {
        upward_equiv[leafs_idx[i]*NSURF+k] = upward_equiv[leafs_idx[i]*NSURF+k]*scal;
      }
    }
  }

  void M2M(Nodes &nodes, RealVec &upward_equiv, std::vector<int> &nonleafs_idx) {
    std::vector<std::vector<int>> nodes_by_level_idx(MAXLEVEL);
    std::vector<std::vector<int>> parent_by_level_idx(MAXLEVEL);
    std::vector<std::vector<int>> octant_by_level_idx(MAXLEVEL);
    for(int i=1;i<nodes.size();i++){
      nodes_by_level_idx[nodes[i].depth-1].push_back(nodes[i].idx);
      parent_by_level_idx[nodes[i].depth-1].push_back(nodes[i].parent->idx);
      octant_by_level_idx[nodes[i].depth-1].push_back(nodes[i].octant);
    }
    M2MGPU(upward_equiv, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx);
  }
  
  void L2L(Node* node, RealVec &dnward_equiv) {
    if(node->IsLeaf()) return;
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        Node* child = node->child[octant];
        RealVec buffer(NSURF);
        gemm(1, NSURF, NSURF, &dnward_equiv[node->idx*NSURF], &(mat_L2L[octant][0]), &buffer[0]);
        for(int k=0; k<NSURF; k++)
          dnward_equiv[child->idx*NSURF+k] += buffer[k];
      }
    }
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        L2L(node->child[octant], dnward_equiv);
    }
    #pragma omp taskwait
  }

  void L2P(Nodes &nodes, RealVec &dnward_equiv, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx) {
    real_t c[3] = {0.0};
    std::vector<RealVec> dnwd_equiv_surf;
    dnwd_equiv_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      dnwd_equiv_surf[depth].resize(NSURF*3);
      dnwd_equiv_surf[depth] = surface(MULTIPOLE_ORDER,c,2.95,depth);
    }
    RealVec equivCoord(leafs_idx.size()*NSURF*3);
    RealVec scal(leafs_idx.size());
    for(int i=0;i<leafs_idx.size(); i++) {
      Node* leaf = &nodes[leafs_idx[i]];
      scal[i] = pow(0.5, leaf->depth);
      for(int k=0; k<NSURF; k++) {
        equivCoord[i*NSURF*3 + 3*k+0] = dnwd_equiv_surf[leaf->depth][3*k+0] + leaf->coord[0];
        equivCoord[i*NSURF*3 + 3*k+1] = dnwd_equiv_surf[leaf->depth][3*k+1] + leaf->coord[1];
        equivCoord[i*NSURF*3 + 3*k+2] = dnwd_equiv_surf[leaf->depth][3*k+2] + leaf->coord[2];
        dnward_equiv[leaf->idx*NSURF+k] *= scal[i];
      }
    }

    //GPUL2P(equivCoord, dnward_equiv, );
    #pragma omp parallel for
    for(int i=0; i<leafs_idx.size(); i++) {
      int leaf_idx = leafs_idx[i];
      Node* leaf = &nodes[leaf_idx];
      // check surface potential -> equivalent surface charge
      RealVec buffer(NSURF);
      gemm(1, NSURF, NSURF, &(dnward_equiv[leaf->idx*NSURF]), &L2L_V[0], &buffer[0]);
      gemm(1, NSURF, NSURF, &buffer[0], &L2L_U[0], &dnward_equiv[leaf->idx*NSURF]);
      // equivalent surface charge -> target potential
      gradientP2P(&equivCoord[i*NSURF*3], NSURF*3, &dnward_equiv[leaf->idx*NSURF], &leaf->pt_coord[0], leaf->pt_coord.size(),  &nodes_trg[nodes_pt_src_idx[leaf_idx]*4]);
    }
  }
    
  void P2L(Nodes& nodes, RealVec &dnward_equiv) {
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
        potentialP2P(&source->pt_coord[0], source->pt_coord.size(), &source->pt_src[0], &targetCheckCoord[0], 3*NSURF, &dnward_equiv[target->idx*NSURF]);
      }
    }
  }
  
  void M2P(Nodes &nodes, std::vector<int>& leafs_idx, RealVec &upward_equiv, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx) {
    real_t c[3] = {0.0};
    std::vector<RealVec> upwd_equiv_surf;
    upwd_equiv_surf.resize(MAXLEVEL+1);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      upwd_equiv_surf[depth].resize(NSURF*3);
      upwd_equiv_surf[depth] = surface(MULTIPOLE_ORDER,c,1.05,depth);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs_idx.size(); i++) {
      Node* target = &nodes[leafs_idx[i]];
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
        gradientP2P(&sourceEquivCoord[0], NSURF*3, &upward_equiv[source->idx*NSURF], &target->pt_coord[0], target->pt_coord.size(), &nodes_trg[nodes_pt_src_idx[leafs_idx[i]]*4]);
      }
    }
  }

  void P2P(Nodes &nodes, std::vector<int> leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx, int ncrit) {
    std::vector<int>P2Plists;
    std::vector<int>P2Plists_idx;
    int P2Plists_idx_cnt = 0;
    
    Profile::Tic("vec to array", true);
    std::vector<int> targets_idx = leafs_idx;
    for(int i=0; i<targets_idx.size(); i++) {
      Node* target = &nodes[targets_idx[i]];
      std::vector<int> sources_idx = target->P2Plist_idx;
      P2Plists.insert(P2Plists.end(), sources_idx.begin(), sources_idx.end());
      P2Plists_idx.push_back(P2Plists_idx_cnt);
      P2Plists_idx_cnt += sources_idx.size();    
    }
    P2Plists_idx.push_back(P2Plists_idx_cnt);
    Profile::Toc();
    std::vector<real_t> trg_val(4*nodes_pt_src_idx[nodes_pt_src_idx.size()-1]);
    P2PGPU(leafs_idx, nodes_coord, nodes_pt_src, nodes_pt_src_idx,P2Plists, P2Plists_idx, trg_val, leafs_idx.size(), ncrit);
    int pt_trg_count = 0;
    Profile::Tic("array to vec", true);
    for(int i=0; i<targets_idx.size(); i++) {
      Node* target = &nodes[targets_idx[i]];
      int trg_size = 4*(nodes_pt_src_idx[target->idx+1]-nodes_pt_src_idx[target->idx]);
      for(int j=0;j< trg_size;j++) {
        nodes_trg[nodes_pt_src_idx[target->idx]*4+j] = trg_val[pt_trg_count++];
      }
    }
    Profile::Toc();
  }

  void hadamardProduct(real_t *kernel, real_t *equiv, real_t *check, int n3_) {
    for(int k=0; k<n3_; ++k) {
      int real = 2*k+0;
      int imag = 2*k+1;
      check[real] += kernel[real]*equiv[real] - kernel[imag]*equiv[imag];
      check[imag] += kernel[real]*equiv[imag] + kernel[imag]*equiv[real];
    }
}

void FFT_UpEquiv(Nodes& nodes, std::vector<int> &M2Lsources_idx, AlignedVec& up_equiv, RealVec &upward_equiv) {
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
    int dim[3] = {n1, n1, n1};
    // evaluate dft of upward equivalent of sources
     #pragma omp parallel for
    for(int i=0; i<M2Lsources_idx.size(); ++i) {
      // upward equiv on convolution grid
      Node *source = &nodes[M2Lsources_idx[i]];
      for(int j=0; j<NSURF; ++j) {
        int conv_id = map[j];
        up_equiv[i*n3+conv_id] = upward_equiv[source->idx*NSURF+j];
      }
    }
  }
  void FFT_Check2Equiv(Nodes& nodes, std::vector<int> &M2Ltargets_idx, std::vector<real_t> dnCheck, RealVec &dnward_equiv) {
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
    
    #pragma omp parallel for
    for(int i=0; i<M2Ltargets_idx.size(); ++i) {
      Node* target = &nodes[M2Ltargets_idx[i]];
      real_t scale = powf(2, target->depth);
      for(int j=0; j<NSURF; ++j) {
        int conv_id = map2[j];
        dnward_equiv[target->idx*NSURF+j] += dnCheck[i*n3+conv_id] * scale;
      }
    }
  }

  void M2L(Nodes& nodes, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, RealVec &upward_equiv, RealVec &dnward_equiv) {
    // define constants
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    AlignedVec up_equiv(M2Lsources_idx.size()*n3);
    std::vector<int> index_in_up_equiv_fft;
    std::vector<int> M2LRelPos_start_idx;
    std::vector<int> M2LRelPoss;
    int M2LRelPos_start_idx_cnt = 0;
    #pragma omp parallel for
    for(int i=0; i<M2Lsources_idx.size(); ++i) {
      Node *node = &nodes[M2Lsources_idx[i]];
      node->index_in_up_equiv_fft = i;
    }
    for (int i=0;i<M2Ltargets_idx.size(); i++) {
      Node* target = &nodes[M2Ltargets_idx[i]];
      M2LRelPos_start_idx.push_back(M2LRelPos_start_idx_cnt);
      for(int j=0; j<target->M2Llist_idx.size(); j++) {
        Node* source = &nodes[target->M2Llist_idx[j]];
        index_in_up_equiv_fft.push_back(source->index_in_up_equiv_fft);
        M2LRelPoss.push_back(target->M2LRelPos[j]);
        M2LRelPos_start_idx_cnt ++;
      }
    }
    M2LRelPos_start_idx.push_back(M2LRelPos_start_idx_cnt);
    FFT_UpEquiv(nodes, M2Lsources_idx, up_equiv, upward_equiv);
    std::vector<real_t> dnCheck = M2LGPU(M2Ltargets_idx, M2LRelPos_start_idx, index_in_up_equiv_fft, M2LRelPoss, mat_M2L_Helper, n3_, up_equiv, M2Lsources_idx.size());
    FFT_Check2Equiv(nodes, M2Ltargets_idx, dnCheck, dnward_equiv);
  }
}//end namespace
