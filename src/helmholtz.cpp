#include "helmholtz.h"

namespace exafmm_t {
  int MULTIPOLE_ORDER;
  int NSURF;
  int MAXLEVEL;
  real_t MU;
  std::vector<M2LData> M2Ldata;

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

  // complex gemm by blas lib
  void gemm(int m, int n, int k, complex_t* A, complex_t* B, complex_t* C) {
    char transA = 'N', transB = 'N';
    complex_t alpha(1., 0.), beta(0.,0.);
#if FLOAT
    cgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    zgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  // complex gemv by blas lib
  void gemv(int m, int n, complex_t* A, complex_t* x, complex_t* y) {
    char trans = 'T';
    complex_t alpha(1., 0.), beta(0.,0.);
    int incx = 1, incy = 1;
#if FLOAT
    cgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    zgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
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

  void svd(int m, int n, complex_t* A, real_t* S, complex_t* U, complex_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    ComplexVec WORK(LWORK);
    RealVec RWORK(5*k);
#if FLOAT
    cgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
#else
    zgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &RWORK[0], &INFO);
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

  ComplexVec transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  ComplexVec conjugate_transpose(ComplexVec& vec, int m, int n) {
    ComplexVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = std::conj(vec[i*n+j]);
      }
    }
    return temp;
  }

  void potential_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    int newton_scale = 16;      // from 2-step Newton method
    const real_t COEF = 1.0/(newton_scale*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    simdvec mu(MU*M_PI/newton_scale);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv_real(zero);
      simdvec tv_imag(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = sx - tx;
        simdvec sy(src_coord[3*s+1]);
        sy = sy - ty;
        simdvec sz(src_coord[3*s+2]);
        sz = sz - tz;
        simdvec sv_real(src_value[s].real());
        simdvec sv_imag(src_value[s].imag());
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;

        simdvec mu_r = mu * r2 * invR;
        simdvec G0 = cos(mu_r)*invR;
        simdvec G1 = sin(mu_r)*invR;
        tv_real += sv_real*G0 - sv_imag*G1;
        tv_imag += sv_real*G1 + sv_imag*G0;
      }
      tv_real *= coef;
      tv_imag *= coef;
      for(int k=0; k<NSIMD && (t+k)<trg_cnt; k++) {
        trg_value[t+k] += complex_t(tv_real[k], tv_imag[k]);
      }
    }
  }

  void gradient_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    simdvec zero((real_t)0);
    int newton_scale = 1;
    for(int i=0; i<2; i++) {
      newton_scale = 2*newton_scale*newton_scale*newton_scale;
    }
    const real_t COEF = 1.0/(newton_scale*4*M_PI);   // factor 16 comes from the simd rsqrt function
    simdvec coef(COEF);
    simdvec mu(MU*M_PI/newton_scale);
    simdvec NEWTON(newton_scale);
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for(int t=0; t<trg_cnt; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv_real(zero);
      simdvec tv_imag(zero);
      simdvec F0_real(zero);
      simdvec F0_imag(zero);
      simdvec F1_real(zero);
      simdvec F1_imag(zero);
      simdvec F2_real(zero);
      simdvec F2_imag(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[3*s+0]);
        sx = sx - tx;
        simdvec sy(src_coord[3*s+1]);
        sy = sy - ty;
        simdvec sz(src_coord[3*s+2]);
        sz = sz - tz;
        simdvec sv_real(src_value[s].real());
        simdvec sv_imag(src_value[s].imag());
        simdvec r2(zero);
        r2 += sx * sx;
        r2 += sy * sy;
        r2 += sz * sz;
        simdvec invR = rsqrt(r2);
        invR &= r2 > zero;

        simdvec mu_r = mu*r2*invR;
        simdvec G0 = cos(mu_r)*invR;
        simdvec G1 = sin(mu_r)*invR;
        simdvec p_real = sv_real*G0 - sv_imag*G1;
        simdvec p_imag = sv_real*G1 + sv_imag*G0;
        tv_real += p_real;
        tv_imag += p_imag;
        simdvec coef_real = invR*invR*p_real/(NEWTON*NEWTON) + mu*p_imag*invR;
        simdvec coef_imag = invR*invR*p_imag/(NEWTON*NEWTON) - mu*p_real*invR;
        F0_real += sx*coef_real;
        F0_imag += sx*coef_imag;
        F1_real += sy*coef_real;
        F1_imag += sy*coef_imag;
        F2_real += sz*coef_real;
        F2_imag += sz*coef_imag;
      }
      tv_real *= coef;
      tv_imag *= coef;
      for(int k=0; k<NSIMD && (t+k)<trg_cnt; k++) {
        trg_value[4*(t+k)+0] += complex_t(tv_real[k], tv_imag[k]);
        trg_value[4*(t+k)+1] += complex_t(F0_real[k], F0_imag[k]);
        trg_value[4*(t+k)+2] += complex_t(F1_real[k], F1_imag[k]);
        trg_value[4*(t+k)+3] += complex_t(F2_real[k], F2_imag[k]);
      }
    }
  }

  // non-simd P2P
#if 0
  void potential_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    complex_t I(0, 1);
    //complex_t WAVEK = complex_t(1,.1) / real_t(2*M_PI);
    real_t WAVEK = 20*M_PI;
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for (int i=0; i<trg_cnt; i++) {
      complex_t p = complex_t(0., 0.);
      cvec3 F = complex_t(0., 0.);
      for (int j=0; j<src_cnt; j++) {
        vec3 dX;
        for (int d=0; d<3; d++) dX[d] = trg_coord[3*i+d] - src_coord[3*j+d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          // real_t R = std::sqrt(R2);
          // newton iteration
          real_t invR = 1 / std::sqrt(R2);  // initial guess
          invR *= 3 - R2*invR*invR;
          invR *= 12 - R2*invR*invR;        // two iterations
          invR /= 16;                       // add back the coefficient in Newton iterations
          real_t R = 1 / invR;
          complex_t pij = std::exp(I * R * WAVEK) * src_value[j] / R;
          complex_t coef = (1/R2 - I*WAVEK/R) * pij;
          p += pij;
        }
      }
      trg_value[i] += p / (4*M_PI);
    }
  }

  void gradient_P2P(RealVec& src_coord, ComplexVec& src_value, RealVec& trg_coord, ComplexVec& trg_value) {
    complex_t I(0, 1);
    //complex_t WAVEK = complex_t(1,.1) / real_t(2*M_PI);
    real_t WAVEK = 20*M_PI;
    int src_cnt = src_coord.size() / 3;
    int trg_cnt = trg_coord.size() / 3;
    for (int i=0; i<trg_cnt; i++) {
      complex_t p = complex_t(0., 0.);
      cvec3 F = complex_t(0., 0.);
      for (int j=0; j<src_cnt; j++) {
        vec3 dX;
        for (int d=0; d<3; d++) dX[d] = trg_coord[3*i+d] - src_coord[3*j+d];
        real_t R2 = norm(dX);
        if (R2 != 0) {
          // real_t R = std::sqrt(R2);
          // newton iteration
          real_t invR = 1 / std::sqrt(R2);  // initial guess
          invR *= 3 - R2*invR*invR;
          invR *= 12 - R2*invR*invR;        // two iterations
          invR /= 16;                       // add back the coefficient in Newton iterations
          real_t R = 1 / invR;
          complex_t pij = std::exp(I * R * WAVEK) * src_value[j] / R;
          complex_t coef = (1/R2 - I*WAVEK/R) * pij;
          p += pij;
          for (int d=0; d<3; d++) {
            F[d] += coef * dX[d];
          }
        }
      }
      trg_value[4*i+0] += p / (4*M_PI);
      trg_value[4*i+1] += F[0] / (4*M_PI);
      trg_value[4*i+2] += F[1] / (4*M_PI);
      trg_value[4*i+3] += F[2] / (4*M_PI);
    }
  }
#endif 

  void kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, complex_t* k_out) {
    ComplexVec src_value(1, complex_t(1,0));
    RealVec trg_coord(r_trg, r_trg+3*trg_cnt);
    #pragma omp parallel for
    for(int i=0; i<src_cnt; i++) {
      RealVec src_coord(r_src+3*i, r_src+3*(i+1));
      ComplexVec trg_value(trg_cnt, complex_t(0,0));
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
      up_check_surf[level] = surface(MULTIPOLE_ORDER,c,2.95,level);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->level;
      RealVec checkCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        checkCoord[3*k+0] = up_check_surf[level][3*k+0] + leaf->xmin[0];
        checkCoord[3*k+1] = up_check_surf[level][3*k+1] + leaf->xmin[1];
        checkCoord[3*k+2] = up_check_surf[level][3*k+2] + leaf->xmin[2];
      }
      potential_P2P(leaf->src_coord, leaf->src_value, checkCoord, leaf->up_equiv);
      ComplexVec buffer(NSURF);
      ComplexVec equiv(NSURF);
      gemv(NSURF, NSURF, &(matrix_UC2E_U[level][0]), &(leaf->up_equiv[0]), &buffer[0]);
      gemv(NSURF, NSURF, &(matrix_UC2E_V[level][0]), &buffer[0], &equiv[0]);
      for(int k=0; k<NSURF; k++)
        leaf->up_equiv[k] = equiv[k];
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
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node* child = node->children[octant];
        ComplexVec buffer(NSURF);
        int level = node->level;
        gemv(NSURF, NSURF, &(matrix_M2M[level][octant][0]), &child->up_equiv[0], &buffer[0]);
        for(int k=0; k<NSURF; k++) {
          node->up_equiv[k] += buffer[k];
        }
      }
    }
  }

  void L2L(Node* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node* child = node->children[octant];
        ComplexVec buffer(NSURF);
        int level = node->level;
        gemv(NSURF, NSURF, &(matrix_L2L[level][octant][0]), &node->dn_equiv[0], &buffer[0]);
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
      dn_equiv_surf[level] = surface(MULTIPOLE_ORDER,c,2.95,level);
    }
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      Node* leaf = leafs[i];
      int level = leaf->level;
      // down check surface potential -> equivalent surface charge
      ComplexVec buffer(NSURF);
      ComplexVec equiv(NSURF);
      gemv(NSURF, NSURF, &(matrix_DC2E_U[level][0]), &(leaf->dn_equiv[0]), &buffer[0]);
      gemv(NSURF, NSURF, &(matrix_DC2E_V[level][0]), &buffer[0], &equiv[0]);
      for(int k=0; k<NSURF; k++)
        leaf->dn_equiv[k] = equiv[k];
      // equivalent surface charge -> target potential
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
      dn_check_surf[level] = surface(MULTIPOLE_ORDER,c,1.05,level);
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
      up_equiv_surf[level] = surface(MULTIPOLE_ORDER,c,1.05,level);
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
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    size_t mat_cnt = rel_coord[M2L_Type].size();
    // initialize M2Ldata
    M2Ldata.resize(MAXLEVEL);
    // construct M2L target nodes for each level
    std::vector<NodePtrs> nodes_out(MAXLEVEL);
    for(int i = 0; i < nonleafs.size(); i++) {
      nodes_out[nonleafs[i]->level].push_back(nonleafs[i]);
    }
    // prepare for M2Ldata for each level
    for(int l = 0; l < MAXLEVEL; l++) {
      // construct M2L source nodes for current level
      std::set<Node*> nodes_in_;
      for(size_t i=0; i<nodes_out[l].size(); i++) {
        NodePtrs& M2L_list = nodes_out[l][i]->M2L_list;
        for(size_t k=0; k<mat_cnt; k++) {
          if(M2L_list[k])
            nodes_in_.insert(M2L_list[k]);
        }
      }
      NodePtrs nodes_in;
      for(std::set<Node*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
        nodes_in.push_back(*node);
      }
      // prepare fft displ
      std::vector<size_t> fft_offset(nodes_in.size());       // displacement in all_up_equiv
      std::vector<size_t> ifft_offset(nodes_out[l].size());  // displacement in all_dn_equiv
      for(size_t i=0; i<nodes_in.size(); i++) {
        fft_offset[i] = nodes_in[i]->children[0]->idx * NSURF;
      }
      for(size_t i=0; i<nodes_out[l].size(); i++) {
        ifft_offset[i] = nodes_out[l][i]->children[0]->idx * NSURF;
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
      size_t fftsize = 2 * 8 * n3;
      for(size_t blk1=0; blk1<n_blk1; blk1++) {
        size_t blk1_start=(nodes_out[l].size()* blk1   )/n_blk1;
        size_t blk1_end  =(nodes_out[l].size()*(blk1+1))/n_blk1;
        for(size_t k=0; k<mat_cnt; k++) {
          for(size_t i=blk1_start; i<blk1_end; i++) {
            NodePtrs& M2L_list = nodes_out[l][i]->M2L_list;
            if(M2L_list[k]) {
              interaction_offset_f.push_back(M2L_list[k]->idx_M2L * fftsize);   // node_in's displacement in fft_in
              interaction_offset_f.push_back(        i           * fftsize);   // node_out's displacement in fft_out
              interaction_count_offset_++;
            }
          }
          interaction_count_offset.push_back(interaction_count_offset_);
        }
      }
      M2Ldata[l].fft_offset     = fft_offset;
      M2Ldata[l].ifft_offset    = ifft_offset;
      M2Ldata[l].interaction_offset_f = interaction_offset_f;
      M2Ldata[l].interaction_count_offset = interaction_count_offset;
    }
  }

  void hadamard_product(std::vector<size_t>& interaction_count_offset, std::vector<size_t>& interaction_offset_f,
                       AlignedVec& fft_in, AlignedVec& fft_out, int level) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    size_t fftsize = 2 * NCHILD * n3;
    AlignedVec zero_vec0(fftsize, 0.);
    AlignedVec zero_vec1(fftsize, 0.);

    // int level = 0;
    size_t mat_cnt = matrix_M2L[level].size();
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
      for(size_t k=0; k<n3; k++) {
        for(size_t mat_indx=0; mat_indx< mat_cnt; mat_indx++) {
          size_t interac_blk1 = blk1*mat_cnt+mat_indx;
          size_t interaction_count_offset0 = (interac_blk1==0?0:interaction_count_offset[interac_blk1-1]);
          size_t interaction_count_offset1 =                    interaction_count_offset[interac_blk1  ] ;
          size_t interac_cnt  = interaction_count_offset1-interaction_count_offset0;
          real_t** IN = &IN_[BLOCK_SIZE*interac_blk1];
          real_t** OUT= &OUT_[BLOCK_SIZE*interac_blk1];
          real_t* M = &matrix_M2L[level][mat_indx][k*2*NCHILD*NCHILD]; // k-th freq's (row) offset in matrix_M2L[mat_indx]
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
    //Profile::Add_FLOP(8*8*8*(interaction_offset_f.size()/2)*n3);
  }

  void fft_up_equiv(std::vector<size_t>& fft_offset, ComplexVec& all_up_equiv, AlignedVec& fft_in) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0, true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER-1-surf[i*3]+0.5))
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+2]+0.5)) * n1 * n1;
    }

    size_t fftsize = 2 * NCHILD * n3;
    ComplexVec fftw_in(n3*NCHILD);
    AlignedVec fftw_out(fftsize);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft(3, dim, NCHILD, reinterpret_cast<fft_complex*>(&fftw_in[0]),
                                      nullptr, 1, n3, (fft_complex*)(&fftw_out[0]), nullptr, 1, n3, 
                                      FFTW_FORWARD, FFTW_ESTIMATE);

    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_offset.size(); node_idx++) {
      RealVec buffer(fftsize, 0);
      ComplexVec equiv_t(NCHILD*n3, complex_t(0.,0.));

      complex_t* up_equiv = &all_up_equiv[fft_offset[node_idx]];  // offset ptr of node's 8 child's up_equiv in all_up_equiv, size=8*NSURF
      real_t* up_equiv_f = &fft_in[fftsize*node_idx];   // offset ptr of node_idx in fft_in vector, size=fftsize

      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<(int)NCHILD; j0++)
          equiv_t[idx+j0*n3] = up_equiv[j0*NSURF+k];
      }
      fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&equiv_t[0]), (fft_complex*)&buffer[0]);
      for(size_t j=0; j<n3; j++) {
        for(size_t k=0; k<NCHILD; k++) {
          up_equiv_f[2*(NCHILD*j+k)+0] = buffer[2*(n3*k+j)+0];
          up_equiv_f[2*(NCHILD*j+k)+1] = buffer[2*(n3*k+j)+1];
        }
      }
    }
    fft_destroy_plan(plan);
  }

  void ifft_dn_check(std::vector<size_t>& ifft_offset, AlignedVec& fft_out, ComplexVec& all_dn_equiv) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0, true);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3]))
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+1])) * n1
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+2])) * n1 * n1;
    }

    size_t fftsize = 2 * NCHILD * n3;
    AlignedVec fftw_in(fftsize);
    ComplexVec fftw_out(n3*NCHILD);
    int dim[3] = {n1, n1, n1};

    fft_plan plan = fft_plan_many_dft(3, dim, NCHILD, (fft_complex*)(&fftw_in[0]), nullptr, 1, n3, 
                                      reinterpret_cast<fft_complex*>(&fftw_out[0]), nullptr, 1, n3, 
                                      FFTW_BACKWARD, FFTW_ESTIMATE);

    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_offset.size(); node_idx++) {
      RealVec buffer0(fftsize, 0);
      ComplexVec buffer1(NCHILD*n3, 0);
      real_t* dn_check_f = &fft_out[fftsize*node_idx];  // offset ptr for node_idx in fft_out vector, size=fftsize
      complex_t* dn_equiv = &all_dn_equiv[ifft_offset[node_idx]];  // offset ptr for node_idx's child's dn_equiv in all_dn_equiv, size=numChilds * NSURF
      for(size_t j=0; j<n3; j++)
        for(size_t k=0; k<NCHILD; k++) {
          buffer0[2*(n3*k+j)+0] = dn_check_f[2*(NCHILD*j+k)+0];
          buffer0[2*(n3*k+j)+1] = dn_check_f[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft(plan, (fft_complex*)&buffer0[0], reinterpret_cast<fft_complex*>(&buffer1[0]));
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dn_equiv[NSURF*j0+k]+=buffer1[idx+j0*n3];
      }
    }
    fft_destroy_plan(plan);
  }
  
  void M2L(Nodes& nodes) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    size_t numNodes = nodes.size();
    ComplexVec all_up_equiv(numNodes*NSURF);
    ComplexVec all_dn_equiv(numNodes*NSURF);
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        all_up_equiv[i*NSURF+j] = nodes[i].up_equiv[j];
        all_dn_equiv[i*NSURF+j] = nodes[i].dn_equiv[j];
      }
    }
    size_t fftsize = 2 * 8 * n3;
    for(int l = 0; l < MAXLEVEL; l++) {
      AlignedVec fft_in(M2Ldata[l].fft_offset.size()*fftsize, 0.);
      AlignedVec fft_out(M2Ldata[l].ifft_offset.size()*fftsize, 0.);
      fft_up_equiv(M2Ldata[l].fft_offset, all_up_equiv, fft_in);
      hadamard_product(M2Ldata[l].interaction_count_offset, M2Ldata[l].interaction_offset_f, fft_in, fft_out, l);
      ifft_dn_check(M2Ldata[l].ifft_offset, fft_out, all_dn_equiv);
    }
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        nodes[i].up_equiv[j] = all_up_equiv[i*NSURF+j];
        nodes[i].dn_equiv[j] = all_dn_equiv[i*NSURF+j];
      }
    }
  }
}//end namespace
