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
    real_t alpha=1.0;
    char transA = 'N', transB = 'N';
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
  
  void L2L(Nodes &nodes, RealVec &dnward_equiv, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx) {
    L2LGPU(nodes, dnward_equiv, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx);
  }

  /*void L2P(Nodes &nodes, RealVec &dnward_equiv, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_coord) {
    real_t c[3] = {0.0};
    std::vector<real_t> dnwd_equiv_surf((MAXLEVEL+1)*NSURF*3);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      surface(MULTIPOLE_ORDER,c,2.95,depth,depth,dnwd_equiv_surf);
    }
    L2PGPU(dnwd_equiv_surf, dnward_equiv, bodies_coord, nodes_trg, leafs_idx, nodes_pt_src_idx);
  }*/

  void L2P(Nodes &nodes, RealVec &dnward_equiv, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_coord) {
    real_t c[3] = {0.0};
    std::vector<real_t> dnwd_equiv_surf((MAXLEVEL+1)*NSURF*3);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      surface(MULTIPOLE_ORDER,c,2.95,depth,depth,dnwd_equiv_surf);
    }
    RealVec equivCoord(leafs_idx.size()*NSURF*3);
    int max = 0;
    for(int i=0;i<leafs_idx.size(); i++) {
      Node* leaf = &nodes[leafs_idx[i]];
      int leaf_idx = leafs_idx[i];
      int node_start = nodes_pt_src_idx[leaf_idx];
      int node_end = nodes_pt_src_idx[leaf_idx+1];
      max = (node_end-node_start)>max?(node_end-node_start):max;
      for(int k=0; k<NSURF; k++) {
        equivCoord[i*NSURF*3 + 3*k+0] = dnwd_equiv_surf[leaf->depth*NSURF*3+3*k+0] + nodes_coord[leaf->idx*3];
        equivCoord[i*NSURF*3 + 3*k+1] = dnwd_equiv_surf[leaf->depth*NSURF*3+3*k+1] + nodes_coord[leaf->idx*3+1];
        equivCoord[i*NSURF*3 + 3*k+2] = dnwd_equiv_surf[leaf->depth*NSURF*3+3*k+2] + nodes_coord[leaf->idx*3+2];
        dnward_equiv[leaf_idx*NSURF+k] *= pow(0.5, leaf->depth);
      }
    }
    L2PGPU(equivCoord, dnward_equiv, bodies_coord, nodes_trg, leafs_idx, nodes_pt_src_idx, max);
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
    RealVec surf(NSURF*3);
    surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0,0,surf);
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
}//end namespace
