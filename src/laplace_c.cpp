#include "laplace_c.h"

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

  // mixed-type gemm: A is complex_t matrix; B is real_t matrix
  void gemm(int m, int n, int k, complex_t* A, real_t* B, complex_t* C) {
    char transA = 'N', transB = 'N';
    complex_t alpha(1., 0.), beta(0.,0.);
#if FLOAT
    scgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dzgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
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

  void potentialP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    for(int i=0; i<trg_coord.size()/3; ++i) {
      complex_t p = 0;
      real_t * tX = &trg_coord[3*i];
      for(int j=0; j<src_value.size(); ++j) {
        vec3 dX;
        real_t * sX = &src_coord[3*j];
        for(int d=0; d<3; ++d) dX[d] = tX[d] - sX[d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          complex_t invR = src_value[j] * sqrt(1.0/R2);
          p += invR;
        }
      }
      trg_value[i] += p;
    }
  }

  void gradientP2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    for(int i=0; i<trg_coord.size()/3; ++i) {
      complex_t p = 0;
      cvec3 F = complex_t(0., 0.);
      real_t * tX = &trg_coord[3*i];
      for(int j=0; j<src_value.size(); ++j) {
        vec3 dX;
        real_t * sX = &src_coord[3*j];
        for(int d=0; d<3; ++d) dX[d] = tX[d] - sX[d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          real_t invR2 = 1.0 / R2;
          complex_t invR = src_value[j] * sqrt(invR2);
          p += invR;
          for(int d=0; d<3; ++d) F[d] += dX[d] * invR2 * invR;
        }
      }
      trg_value[4*i+0] += p;
      trg_value[4*i+1] += F[0];
      trg_value[4*i+2] += F[1];
      trg_value[4*i+3] += F[2];
    }
  }

  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    RealVec src_value(1, 1);
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      RealVec trg_value(trg_cnt, 0);
      potentialP2P(src_coord, src_value, trg_coord, trg_value);
      std::copy(trg_value.begin(), trg_value.end(), &k_out[i*trg_cnt]);
    }
  }
 
  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out) {
    ComplexVec src_value(1, complex_t(1,0));
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      ComplexVec trg_value(trg_cnt, complex_t(0,0));
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
      ComplexVec buffer(NSURF);
      ComplexVec equiv(NSURF);
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
        ComplexVec buffer(NSURF);
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
        ComplexVec buffer(NSURF);
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
      ComplexVec buffer(NSURF);
      ComplexVec equiv(NSURF);
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
      if (target->IsLeaf() && target->numBodies<=NSURF)
        continue;
      std::vector<Node*>& sources = target->interac_list[P2L_Type];
      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
        if (source != NULL) {
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
  }

  void M2P(std::vector<Node*>& leafs) {
    std::vector<Node*>& targets = leafs;  // leafs
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
      std::vector<Node*>& sources = target->interac_list[M2P_Type];
      for(int j=0; j<sources.size(); j++) {
        Node* source = sources[j];
        if (source != NULL) {
          if (source->IsLeaf() && source->numBodies<=NSURF)
            continue;
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
  }

  void P2P(std::vector<Node*>& leafs) {
    std::vector<Node*>& targets = leafs;   // leafs, assume sources == targets
    std::vector<Mat_Type> types = {P2P0_Type, P2P1_Type, P2P2_Type, P2L_Type, M2P_Type};
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      Node* target = targets[i];
      for(int k=0; k<types.size(); k++) {
        Mat_Type type = types[k];
        std::vector<Node*>& sources = target->interac_list[type];
        if (type == P2L_Type)
          if (target->numBodies > NSURF) {
            continue;
          }
        for(int j=0; j<sources.size(); j++) {
          Node* source = sources[j];
          if (source != NULL) {
            if (type == M2P_Type) {
              if (source->numBodies > NSURF) {
                continue;
              }
            }
            gradientP2P(source->pt_coord, source->pt_src, target->pt_coord, target->pt_trg);
          }
        }
      }
    }
  }
}//end namespace
