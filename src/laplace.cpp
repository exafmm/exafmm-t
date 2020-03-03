#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "laplace.h"

namespace exafmm_t {
  //! blas gemv with row major data
  void LaplaceFMM::gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
    char trans = 'T';
    real_t alpha = 1.0, beta = 0.0;
    int incx = 1, incy = 1;
#if FLOAT
    sgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#else
    dgemv_(&trans, &n, &m, &alpha, A, &n, x, &incx, &beta, y, &incy);
#endif
  }

  //! blas gemm with row major data
  void LaplaceFMM::gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void LaplaceFMM::svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
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

  RealVec LaplaceFMM::transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void LaplaceFMM::potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero(real_t(0));
    real_t newton_coef = 16;   // comes from Newton's method in simd rsqrt function
    const real_t COEF = 1.0/(4*PI*newton_coef);
    simdvec coef(COEF);
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    int t;
    for(t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv(zero);
      for(int s=0; s<nsrcs; s++) {
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
      for(int k=0; k<NSIMD && t+k<ntrgs; k++) {
        trg_value[t+k] += tv[k];
      }
    }
    for(; t<ntrgs; t++) {
      real_t potential = 0;
      for(int s=0; s<nsrcs; ++s) {
        vec3 dx = 0;
        for(int d=0; d<3; d++) {
          dx[d] += trg_coord[3*t+d] - src_coord[3*s+d];
        }
        real_t r2 = norm(dx);
        if(r2!=0) {
          real_t inv_r = 1 / std::sqrt(r2);
          potential += src_value[s] * inv_r;
        }
      }
      trg_value[t] += potential / (4*PI);
    }
  }

  void LaplaceFMM::gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    simdvec zero(real_t(0));
    real_t newton_coefp = 16;   // comes from Newton's method in simd rsqrt function
    real_t newton_coefg = 16*16*16;
    const real_t COEFP = 1.0/(4*PI*newton_coefp);
    const real_t COEFG = -1.0/(4*PI*newton_coefg);
    simdvec coefp(COEFP);
    simdvec coefg(COEFG);
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    int t;
    for(t=0; t+NSIMD<=ntrgs; t+=NSIMD) {
      simdvec tx(&trg_coord[3*t+0], 3*(int)sizeof(real_t));
      simdvec ty(&trg_coord[3*t+1], 3*(int)sizeof(real_t));
      simdvec tz(&trg_coord[3*t+2], 3*(int)sizeof(real_t));
      simdvec tv0(zero);
      simdvec tv1(zero);
      simdvec tv2(zero);
      simdvec tv3(zero);
      for(int s=0; s<nsrcs; s++) {
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
      for(int k=0; k<NSIMD && t+k<ntrgs; k++) {
        trg_value[0+4*(t+k)] += tv0[k];
        trg_value[1+4*(t+k)] += tv1[k];
        trg_value[2+4*(t+k)] += tv2[k];
        trg_value[3+4*(t+k)] += tv3[k];
      }
    }
    for(; t<ntrgs; t++) {
      real_t potential = 0;
      vec3 gradient = 0;
      for(int s=0; s<nsrcs; ++s) {
        vec3 dx = 0;
        for (int d=0; d<3; ++d) {
          dx[d] = trg_coord[3*t+d] - src_coord[3*s+d];
        }
        real_t r2 = norm(dx);
        if (r2!=0) {
          real_t inv_r2 = 1.0 / r2;
          real_t inv_r = src_value[s] * std::sqrt(inv_r2);
          potential += inv_r;
          dx *= inv_r2 * inv_r;
          gradient[0] += dx[0];
          gradient[1] += dx[1];
          gradient[2] += dx[2];
        }
      }
      trg_value[4*t] += potential / (4*PI) ;
      trg_value[4*t+1] -= gradient[0] / (4*PI);
      trg_value[4*t+2] -= gradient[1] / (4*PI);
      trg_value[4*t+3] -= gradient[2] / (4*PI);
    }
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  void LaplaceFMM::kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
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

  void LaplaceFMM::initialize_matrix() {
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    matrix_UC2E_U.resize(nsurf*nsurf);
    matrix_UC2E_V.resize(nsurf*nsurf);
    matrix_DC2E_U.resize(nsurf*nsurf);
    matrix_DC2E_V.resize(nsurf*nsurf);
    matrix_M2M.resize(REL_COORD[M2M_Type].size(), RealVec(nsurf*nsurf));
    matrix_L2L.resize(REL_COORD[L2L_Type].size(), RealVec(nsurf*nsurf));
    matrix_M2L.resize(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
  }

  void LaplaceFMM::precompute_check2equiv() {
    int level = 0;
    real_t c[3] = {0, 0, 0};
    // caculate upwardcheck to equiv U and V
    RealVec up_check_surf = surface(p, r0, level, c, 2.95);
    RealVec up_equiv_surf = surface(p, r0, level, c, 1.05);
    RealVec matrix_c2e(nsurf*nsurf);
    kernel_matrix(&up_check_surf[0], nsurf, &up_equiv_surf[0], nsurf, &matrix_c2e[0]);
    RealVec U(nsurf*nsurf), S(nsurf*nsurf), VT(nsurf*nsurf);
    svd(nsurf, nsurf, &matrix_c2e[0], &S[0], &U[0], &VT[0]);
    // inverse S
    real_t max_S = 0;
    for(int i=0; i<nsurf; i++) {
      max_S = fabs(S[i*nsurf+i])>max_S ? fabs(S[i*nsurf+i]) : max_S;
    }
    for(int i=0; i<nsurf; i++) {
      S[i*nsurf+i] = S[i*nsurf+i]>EPS*max_S*4 ? 1.0/S[i*nsurf+i] : 0.0;
    }
    // save matrix
    RealVec V = transpose(VT, nsurf, nsurf);
    matrix_UC2E_U = transpose(U, nsurf, nsurf);
    gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &matrix_UC2E_V[0]);

    matrix_DC2E_U = VT;
    gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &matrix_DC2E_V[0]);
  }

  void LaplaceFMM::precompute_M2M() {
    int numRelCoord = REL_COORD[M2M_Type].size();
    int level = 0;
    real_t parent_coord[3] = {0, 0, 0};
    RealVec parent_up_check_surf = surface(p, r0, level, parent_coord, 2.95);
    real_t s = r0 * powf(0.5, level+1);
#pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      ivec3& coord = REL_COORD[M2M_Type][i];
      real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                               parent_coord[1] + coord[1]*s,
                               parent_coord[2] + coord[2]*s};
      RealVec child_up_equiv_surf = surface(p, r0, level+1, child_coord, 1.05);
      RealVec matrix_pc2ce(nsurf*nsurf);
      kernel_matrix(&parent_up_check_surf[0], nsurf, &child_up_equiv_surf[0], nsurf, &matrix_pc2ce[0]);
      // M2M
      RealVec buffer(nsurf*nsurf);
      gemm(nsurf, nsurf, nsurf, &matrix_UC2E_U[0], &matrix_pc2ce[0], &buffer[0]);
      gemm(nsurf, nsurf, nsurf, &matrix_UC2E_V[0], &buffer[0], &(matrix_M2M[i][0]));
      // L2L
      matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &matrix_DC2E_V[0], &buffer[0]);
      gemm(nsurf, nsurf, nsurf, &buffer[0], &matrix_DC2E_U[0], &(matrix_L2L[i][0]));
    }
  }

  void LaplaceFMM::precompute_M2L(std::vector<std::vector<int>>& parent2child) {
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);

    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(),
                                           RealVec(2*n3_));
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*p, 2*p, 2*p};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, fftw_in.data(), nullptr, 1, n3,
                    reinterpret_cast<fft_complex*>(fftw_out.data()), nullptr, 1, n3_,
                    FFTW_ESTIMATE);
    // Precompute M2L matrix
    RealVec trg_coord(3,0);
    // compute DFT of potentials at convolution grids
#pragma omp parallel for
    for(size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); ++i) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = REL_COORD[M2L_Helper_Type][i][d] * r0 / 0.5;
      }
      RealVec conv_coord = convolution_grid(p, r0, 0, coord);   // convolution grid
      RealVec conv_p(n3);   // potentials on convolution grid
      kernel_matrix(conv_coord.data(), n3, trg_coord.data(), 1, conv_p.data());
      fft_execute_dft_r2c(plan, conv_p.data(), 
          reinterpret_cast<fft_complex*>(matrix_M2L_Helper[i].data()));
    }
    // convert M2L_Helper to M2L, reorder to improve data locality
#pragma omp parallel for
    for(size_t i=0; i<REL_COORD[M2L_Type].size(); ++i) {
      for(int j=0; j<NCHILD*NCHILD; j++) {   // loop over child's relative positions
        int child_rel_idx = parent2child[i][j];
        if (child_rel_idx != -1) {
          for(int k=0; k<n3_; k++) {   // loop over frequencies
            int new_idx = k*(2*NCHILD*NCHILD) + 2*j;
            matrix_M2L[i][new_idx+0] = matrix_M2L_Helper[child_rel_idx][k*2+0] / n3;   // real
            matrix_M2L[i][new_idx+1] = matrix_M2L_Helper[child_rel_idx][k*2+1] / n3;   // imag
          }
        }
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

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

  void LaplaceFMM::save_matrix() {
    std::remove(filename.c_str());
    std::ofstream file(filename, std::ofstream::binary);
    // r0
    file.write(reinterpret_cast<char*>(&r0), sizeof(real_t));
    size_t size = nsurf*nsurf;
    // UC2E, DC2E
    file.write(reinterpret_cast<char*>(&matrix_UC2E_U[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_UC2E_V[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_DC2E_U[0]), size*sizeof(real_t));
    file.write(reinterpret_cast<char*>(&matrix_DC2E_V[0]), size*sizeof(real_t));
    // M2M, L2L
    for(auto & vec : matrix_M2M) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    for(auto & vec : matrix_L2L) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    // M2L
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    size = n3_ * 2 * NCHILD * NCHILD;
    for(auto & vec : matrix_M2L) {
      file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
    }
    file.close();
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

  void LaplaceFMM::P2L(Nodes_t& nodes) {
    Nodes_t& targets = nodes;
    real_t c[3] = {0.0};
    std::vector<RealVec> dn_check_surf;
    dn_check_surf.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      dn_check_surf[level].resize(nsurf*3);
      dn_check_surf[level] = surface(p, r0, level, c, 1.05);
    }
    #pragma omp parallel for
    for(size_t i=0; i<targets.size(); i++) {
      Node_t* target = &targets[i];
      NodePtrs_t& sources = target->P2L_list;
      for(size_t j=0; j<sources.size(); j++) {
        Node_t* source = sources[j];
        RealVec trg_check_coord(nsurf*3);
        int level = target->level;
        // target node's check coord = relative check coord + node's origin
        for(int k=0; k<nsurf; k++) {
          trg_check_coord[3*k+0] = dn_check_surf[level][3*k+0] + target->x[0];
          trg_check_coord[3*k+1] = dn_check_surf[level][3*k+1] + target->x[1];
          trg_check_coord[3*k+2] = dn_check_surf[level][3*k+2] + target->x[2];
        }
        potential_P2P(source->src_coord, source->src_value, trg_check_coord, target->dn_equiv);
      }
    }
  }

  void LaplaceFMM::M2P(NodePtrs_t& leafs) {
    NodePtrs_t& targets = leafs;
    real_t c[3] = {0.0};
    std::vector<RealVec> up_equiv_surf;
    up_equiv_surf.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      up_equiv_surf[level].resize(nsurf*3);
      up_equiv_surf[level] = surface(p, r0, level, c, 1.05,level);
    }
    #pragma omp parallel for
    for(size_t i=0; i<targets.size(); i++) {
      Node_t* target = targets[i];
      NodePtrs_t& sources = target->M2P_list;
      for(size_t j=0; j<sources.size(); j++) {
        Node_t* source = sources[j];
        RealVec sourceEquivCoord(nsurf*3);
        int level = source->level;
        // source node's equiv coord = relative equiv coord + node's origin
        for(int k=0; k<nsurf; k++) {
          sourceEquivCoord[3*k+0] = up_equiv_surf[level][3*k+0] + source->x[0];
          sourceEquivCoord[3*k+1] = up_equiv_surf[level][3*k+1] + source->x[1];
          sourceEquivCoord[3*k+2] = up_equiv_surf[level][3*k+2] + source->x[2];
        }
        gradient_P2P(sourceEquivCoord, source->up_equiv, target->trg_coord, target->trg_value);
      }
    }
  }

  void LaplaceFMM::P2P(NodePtrs_t& leafs) {
    NodePtrs_t& targets = leafs;   // assume sources == targets
    #pragma omp parallel for
    for(size_t i=0; i<targets.size(); i++) {
      Node_t* target = targets[i];
      NodePtrs_t& sources = target->P2P_list;
      for(size_t j=0; j<sources.size(); j++) {
        Node_t* source = sources[j];
        gradient_P2P(source->src_coord, source->src_value, target->trg_coord, target->trg_value);
      }
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

   void LaplaceFMM::upward_pass(Nodes_t& nodes, NodePtrs_t& leafs) {
    start("P2M");
    P2M(leafs);
    stop("P2M");
    start("M2M");
    #pragma omp parallel
    #pragma omp single nowait
    M2M(&nodes[0]);
    stop("M2M");
  }

  void LaplaceFMM::downward_pass(Nodes_t& nodes, NodePtrs_t& leafs) {
    start("P2L");
    P2L(nodes);
    stop("P2L");
    start("M2P");
    M2P(leafs);
    stop("M2P");
    start("P2P");
    P2P(leafs);
    stop("P2P");
    start("M2L");
    M2L(nodes);
    stop("M2L");
    start("L2L");
    #pragma omp parallel
    #pragma omp single nowait
    L2L(&nodes[0]);
    stop("L2L");
    start("L2P");
    L2P(leafs);
    stop("L2P");
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
