#include <cstring>  // std::memset()
#include <fstream>  // std::ifstream
#include <set>      // std::set
#include "modified_helmholtz.h"

namespace exafmm_t {
  //! blas gemv with row major data
  void ModifiedHelmholtzFMM::gemv(int m, int n, real_t* A, real_t* x, real_t* y) {
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
  void ModifiedHelmholtzFMM::gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void ModifiedHelmholtzFMM::svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
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

  RealVec ModifiedHelmholtzFMM::transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

  void ModifiedHelmholtzFMM::potential_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    for (int i=0; i<ntrgs; ++i) {
      vec3 x_trg;
      real_t potential = 0;
      for (int d=0; d<3; ++d)
        x_trg[d] = trg_coord[3*i+d];
      for (int j=0; j<nsrcs; ++j) {
        vec3 x_src;
        for (int d=0; d<3; ++d) {
          x_src[d] = src_coord[3*j+d];
        }
        vec3 dx = x_trg - x_src;
        real_t r = std::sqrt(norm(dx));
        if (r>0) {
          potential += std::exp(-wavek*r) / r * src_value[j];
        }
      }
      trg_value[i] += potential / 4*PI;
    }
  }

  void ModifiedHelmholtzFMM::gradient_P2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
    int nsrcs = src_coord.size() / 3;
    int ntrgs = trg_coord.size() / 3;
    for (int i=0; i<ntrgs; ++i) {
      vec3 x_trg;
      real_t potential = 0;
      vec3 gradient = 0;
      for (int d=0; d<3; ++d)
        x_trg[d] = trg_coord[3*i+d];
      for (int j=0; j<nsrcs; ++j) {
        vec3 x_src;
        for (int d=0; d<3; ++d) {
          x_src[d] = src_coord[3*j+d];
        }
        vec3 dx = x_trg - x_src;
        real_t r = std::sqrt(norm(dx));
        // dp / dr
        if (r>0) {
          real_t kernel  = std::exp(-wavek*r) / r;
          real_t dpdr = - kernel * (wavek*r+1) / r;
          potential += kernel * src_value[j];
          gradient[0] += dpdr / r * dx[0] * src_value[j];
          gradient[1] += dpdr / r * dx[1] * src_value[j];
          gradient[2] += dpdr / r * dx[2] * src_value[j];
        }
      }
      trg_value[4*i+0] += potential / 4*PI;
      trg_value[4*i+1] += gradient[0] / 4*PI;
      trg_value[4*i+2] += gradient[1] / 4*PI;
      trg_value[4*i+3] += gradient[2] / 4*PI;
    }
  }

  //! P2P save pairwise contributions to k_out (not aggregate over each target)
  void ModifiedHelmholtzFMM::kernel_matrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
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

  void ModifiedHelmholtzFMM::initialize_matrix() {
    matrix_UC2E_V.resize(depth+1);
    matrix_UC2E_U.resize(depth+1);
    matrix_DC2E_V.resize(depth+1);
    matrix_DC2E_U.resize(depth+1);
    matrix_M2M.resize(depth+1);
    matrix_L2L.resize(depth+1);
    for(int level = 0; level <= depth; level++) {
      matrix_UC2E_V[level].resize(nsurf*nsurf);
      matrix_UC2E_U[level].resize(nsurf*nsurf);
      matrix_DC2E_V[level].resize(nsurf*nsurf);
      matrix_DC2E_U[level].resize(nsurf*nsurf);
      matrix_M2M[level].resize(REL_COORD[M2M_Type].size(), RealVec(nsurf*nsurf));
      matrix_L2L[level].resize(REL_COORD[L2L_Type].size(), RealVec(nsurf*nsurf));
    }
  }

  void ModifiedHelmholtzFMM::precompute_check2equiv() {
    real_t c[3] = {0, 0, 0};
    for(int level = 0; level <= depth; level++) {
      // caculate matrix_UC2E_U and matrix_UC2E_V
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
      matrix_UC2E_U[level] = transpose(U, nsurf, nsurf);
      gemm(nsurf, nsurf, nsurf, &V[0], &S[0], &(matrix_UC2E_V[level][0]));

      matrix_DC2E_U[level] = VT;
      gemm(nsurf, nsurf, nsurf, &U[0], &S[0], &(matrix_DC2E_V[level][0]));
    }
  }

  void ModifiedHelmholtzFMM::precompute_M2M() {
    real_t parent_coord[3] = {0, 0, 0};
    for(int level = 0; level <= depth; level++) {
      RealVec parent_up_check_surf = surface(p, r0, level, parent_coord, 2.95);
      real_t s = r0 * powf(0.5, level+1);
      int num_coords = REL_COORD[M2M_Type].size();
#pragma omp parallel for
      for(int i=0; i<num_coords; i++) {
        ivec3& coord = REL_COORD[M2M_Type][i];
        real_t child_coord[3] = {parent_coord[0] + coord[0]*s,
                                 parent_coord[1] + coord[1]*s,
                                 parent_coord[2] + coord[2]*s};
        RealVec child_up_equiv_surf = surface(p, r0, level+1, child_coord, 1.05);
        RealVec matrix_pc2ce(nsurf*nsurf);
        kernel_matrix(&parent_up_check_surf[0], nsurf, &child_up_equiv_surf[0], nsurf, &matrix_pc2ce[0]);
        // M2M: child's upward_equiv to parent's check
        RealVec buffer(nsurf*nsurf);
        gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_U[level][0]), &matrix_pc2ce[0], &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &(matrix_M2M[level][i][0]));
        // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
        matrix_pc2ce = transpose(matrix_pc2ce, nsurf, nsurf);
        gemm(nsurf, nsurf, nsurf, &matrix_pc2ce[0], &(matrix_DC2E_V[level][0]), &buffer[0]);
        gemm(nsurf, nsurf, nsurf, &buffer[0], &(matrix_DC2E_U[level][0]), &(matrix_L2L[level][i][0]));
      }
    }
  }

  void ModifiedHelmholtzFMM::precompute_M2L(std::ofstream& file, std::vector<std::vector<int>>& parent2child) {
    int n1 = p * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    int fft_size = 2 * n3_ * NCHILD * NCHILD;

    std::vector<RealVec> matrix_M2L_Helper(REL_COORD[M2L_Helper_Type].size(), RealVec(2*n3_));
    std::vector<AlignedVec> matrix_M2L(REL_COORD[M2L_Type].size(), AlignedVec(fft_size));
    // create fftw plan
    RealVec fftw_in(n3);
    RealVec fftw_out(2*n3_);
    int dim[3] = {2*p, 2*p, 2*p};
    fft_plan plan = fft_plan_many_dft_r2c(3, dim, 1, fftw_in.data(), nullptr, 1, n3, 
                    reinterpret_cast<fft_complex*>(fftw_out.data()), nullptr, 1, n3_,
                    FFTW_ESTIMATE);
    // Precompute M2L matrix
    RealVec trg_coord(3,0);
    for(int l=1; l<depth+1; ++l) {
      // compute DFT of potentials at convolution grids
#pragma omp parallel for
      for(size_t i=0; i<REL_COORD[M2L_Helper_Type].size(); i++) {
        real_t coord[3];
        for(int d=0; d<3; d++) {
          coord[d] = REL_COORD[M2L_Helper_Type][i][d] * r0 * powf(0.5, l-1);
        }
        RealVec conv_coord = convolution_grid(p, r0, l, coord);   // convolution grid
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
      // write to file
      for(auto& vec : matrix_M2L) {
        file.write(reinterpret_cast<char*>(vec.data()), fft_size*sizeof(real_t));
      }
    }
    // destroy fftw plan
    fft_destroy_plan(plan);
  }

  bool ModifiedHelmholtzFMM::load_matrix() {
    std::ifstream file(filename, std::ifstream::binary);
    int n1 = p * 2;
    int n3_ = n1 * n1 * (n1/2 + 1);
    size_t fft_size = n3_ * 2 * NCHILD * NCHILD;
    size_t file_size = (2*REL_COORD[M2M_Type].size()+4) * nsurf * nsurf * (depth+1) * sizeof(real_t)
                     +  REL_COORD[M2L_Type].size() * fft_size * depth * sizeof(real_t) + 2 * sizeof(real_t);    // 2 denotes r0 and wavek
    if (file.good()) {     // if file exists
      file.seekg(0, file.end);
      if (size_t(file.tellg()) == file_size) {   // if file size is correct
        file.seekg(0, file.beg);         // move the position back to the beginning
        // check whether r0 matches
        real_t r0_;
        file.read(reinterpret_cast<char*>(&r0_), sizeof(real_t));
        if (r0 != r0_) {
          std::cout << r0 << " " << r0_ << std::endl;
          return false;
        }
        // check whether wavek matches
        real_t wavek_;
        file.read(reinterpret_cast<char*>(&wavek_), sizeof(real_t));
        if (wavek != wavek_) {
          std::cout << wavek << " " << wavek_ << std::endl;
          return false;
        }
        size_t size = nsurf * nsurf;
        for(int level = 0; level <= depth; level++) {
          // UC2E, DC2E
          file.read(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(real_t));
          file.read(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(real_t));
          // M2M, L2L
          for(auto & vec : matrix_M2M[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
          }
          for(auto & vec : matrix_L2L[level]) {
            file.read(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
          }
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

  void ModifiedHelmholtzFMM::save_matrix(std::ofstream& file) {
    // root radius r0
    file.write(reinterpret_cast<char*>(&r0), sizeof(real_t));
    // wavenumber wavek
    file.write(reinterpret_cast<char*>(&wavek), sizeof(real_t));
    size_t size = nsurf*nsurf;
    for(int level = 0; level <= depth; level++) {
      // save UC2E, DC2E precompute data
      file.write(reinterpret_cast<char*>(&matrix_UC2E_U[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_UC2E_V[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_U[level][0]), size*sizeof(real_t));
      file.write(reinterpret_cast<char*>(&matrix_DC2E_V[level][0]), size*sizeof(real_t));
      // M2M, M2L precompute data
      for(auto & vec : matrix_M2M[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
      for(auto & vec : matrix_L2L[level]) {
        file.write(reinterpret_cast<char*>(&vec[0]), size*sizeof(real_t));
      }
    }
  }

  void ModifiedHelmholtzFMM::precompute() {
    // if matrix binary file exists
    filename = "modified_helmholtz";
    filename += "_" + std::string(sizeof(real_t)==4 ? "f":"d") + "_" + "p" + std::to_string(p) + "_" + "l" + std::to_string(depth);
    filename += ".dat";
    initialize_matrix();
    if (load_matrix()) {
      is_precomputed = false;
      return;
    } else {
      precompute_check2equiv();
      precompute_M2M();
      std::remove(filename.c_str());
      std::ofstream file(filename, std::ofstream::binary);
      save_matrix(file);
      auto parent2child = map_matrix_index();
      precompute_M2L(file, parent2child);
      file.close();
    }
  }

  void ModifiedHelmholtzFMM::P2M(NodePtrs_t& leafs) {
    real_t c[3] = {0, 0, 0};
    std::vector<RealVec> up_check_surf;
    up_check_surf.resize(depth+1);
    for(int level=0; level<=depth; level++) {
      up_check_surf[level].resize(nsurf*3);
      up_check_surf[level] = surface(p, r0, level, c, 2.95);
    }
    #pragma omp parallel for
    for(size_t i=0; i<leafs.size(); i++) {
      Node_t* leaf = leafs[i];
      int level = leaf->level;
      RealVec checkCoord(nsurf*3);
      for(int k=0; k<nsurf; k++) {
        checkCoord[3*k+0] = up_check_surf[level][3*k+0] + leaf->x[0];
        checkCoord[3*k+1] = up_check_surf[level][3*k+1] + leaf->x[1];
        checkCoord[3*k+2] = up_check_surf[level][3*k+2] + leaf->x[2];
      }
      potential_P2P(leaf->src_coord, leaf->src_value, checkCoord, leaf->up_equiv);
      RealVec buffer(nsurf);
      RealVec equiv(nsurf);
      gemv(nsurf, nsurf, &(matrix_UC2E_U[level][0]), &(leaf->up_equiv[0]), &buffer[0]);
      gemv(nsurf, nsurf, &(matrix_UC2E_V[level][0]), &buffer[0], &equiv[0]);
      for(int k=0; k<nsurf; k++)
        leaf->up_equiv[k] = equiv[k];
    }
  }

  void ModifiedHelmholtzFMM::M2M(Node_t* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant])
        #pragma omp task untied
        M2M(node->children[octant]);
    }
    #pragma omp taskwait
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node_t* child = node->children[octant];
        RealVec buffer(nsurf);
        int level = node->level;
        gemv(nsurf, nsurf, &(matrix_M2M[level][octant][0]), &child->up_equiv[0], &buffer[0]);
        for(int k=0; k<nsurf; k++) {
          node->up_equiv[k] += buffer[k];
        }
      }
    }
  }

  void ModifiedHelmholtzFMM::L2L(Node_t* node) {
    if(node->is_leaf) return;
    for(int octant=0; octant<8; octant++) {
      if(node->children[octant]) {
        Node_t* child = node->children[octant];
        RealVec buffer(nsurf);
        int level = node->level;
        gemv(nsurf, nsurf, &(matrix_L2L[level][octant][0]), &node->dn_equiv[0], &buffer[0]);
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

  void ModifiedHelmholtzFMM::L2P(NodePtrs_t& leafs) {
    real_t c[3] = {0, 0, 0};
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
      // down check surface potential -> equivalent surface charge
      RealVec buffer(nsurf);
      RealVec equiv(nsurf);
      gemv(nsurf, nsurf, &(matrix_DC2E_U[level][0]), &(leaf->dn_equiv[0]), &buffer[0]);
      gemv(nsurf, nsurf, &(matrix_DC2E_V[level][0]), &buffer[0], &equiv[0]);
      for(int k=0; k<nsurf; k++)
        leaf->dn_equiv[k] = equiv[k];
      // equivalent surface charge -> target potential
      RealVec equivCoord(nsurf*3);
      for(int k=0; k<nsurf; k++) {
        equivCoord[3*k+0] = dn_equiv_surf[level][3*k+0] + leaf->x[0];
        equivCoord[3*k+1] = dn_equiv_surf[level][3*k+1] + leaf->x[1];
        equivCoord[3*k+2] = dn_equiv_surf[level][3*k+2] + leaf->x[2];
      }
      gradient_P2P(equivCoord, leaf->dn_equiv, leaf->trg_coord, leaf->trg_value);
    }
  }

  void ModifiedHelmholtzFMM::P2L(Nodes_t& nodes) {
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

  void ModifiedHelmholtzFMM::M2P(NodePtrs_t& leafs) {
    NodePtrs_t& targets = leafs;
    real_t c[3] = {0.0};
    std::vector<RealVec> up_equiv_surf;
    up_equiv_surf.resize(depth+1);
    for(int level=0; level<=depth; level++) {
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

  void ModifiedHelmholtzFMM::P2P(NodePtrs_t& leafs) {
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

   void ModifiedHelmholtzFMM::upward_pass(Nodes_t& nodes, NodePtrs_t& leafs) {
    start("P2M");
    P2M(leafs);
    stop("P2M");
    start("M2M");
    #pragma omp parallel
    #pragma omp single nowait
    M2M(&nodes[0]);
    stop("M2M");
  }

  void ModifiedHelmholtzFMM::downward_pass(Nodes_t& nodes, NodePtrs_t& leafs) {
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

  RealVec ModifiedHelmholtzFMM::verify(NodePtrs_t& leafs) {
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
