#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_
#include "geometry.h"
#include "intrinsics.h"
#include "pvfmm.h"

namespace pvfmm {
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
    simdvec zero((real_t)0);
    const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);
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
  // For Laplace: ker_dim[0] = 1, j = 0; Force a unit charge (q=1)
  // r_src layout: [x1, y1, z1, x2, y2, z2, ...]
  // k_out layout (potential): [p11, p12, p13, ..., p21, p22, ...]  (1st digit: src_idx; 2nd: trg_idx)
  // k_out layout (gradient) : [Fx11, Fy11, Fz11, Fx12, Fy12, Fz13, ... Fx1n, Fy1n, Fz1n, ...
  //                            Fx21, Fy21, Fz21, Fx22, Fy22, Fz22, ... Fx2n, Fy2n, Fz2n, ...
  //                            ...]
  void BuildMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
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

  void P2M() {
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);    // scaling factor of UC2UE precomputation matrix source charge -> check surface potential
      std::vector<real_t> checkCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        checkCoord[3*k+0] = upwd_check_surf[level][3*k+0] + leaf->coord[0];
        checkCoord[3*k+1] = upwd_check_surf[level][3*k+1] + leaf->coord[1];
        checkCoord[3*k+2] = upwd_check_surf[level][3*k+2] + leaf->coord[2];
      }
      potentialP2P(leaf->pt_coord, leaf->pt_src, checkCoord, leaf->upward_equiv);
      Matrix<real_t> check(1, NSURF, &(leaf->upward_equiv[0]));  // check surface potential
      Matrix<real_t> buffer(1, NSURF);
      Matrix<real_t>::GEMM(buffer, check, M2M_V);
      Matrix<real_t> equiv(1, NSURF);  // equivalent surface charge
      Matrix<real_t>::GEMM(equiv, buffer, M2M_U);
      for(int k=0; k<NSURF; k++)
        leaf->upward_equiv[k] = scal * equiv[0][k];
    }
  }

  void M2M(FMM_Node* node) {
    if(node->IsLeaf()) return;
    Matrix<real_t>& M = mat_M2M;
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        M2M(node->child[octant]);
    }
    #pragma omp taskwait
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        FMM_Node* child = node->child[octant];
        std::vector<size_t>& perm_in = perm_r[octant].perm;
        std::vector<size_t>& perm_out = perm_c[octant].perm;
        Matrix<real_t> buffer_in(1, NSURF);
        Matrix<real_t> buffer_out(1, NSURF);
        for(int k=0; k<NSURF; k++) {
          buffer_in[0][k] = child->upward_equiv[perm_in[k]]; // input perm
        }
        Matrix<real_t>::GEMM(buffer_out, buffer_in, M);
        for(int k=0; k<NSURF; k++)
          node->upward_equiv[k] += buffer_out[0][perm_out[k]];
      }
    }
  }

  void L2L(FMM_Node* node) {
    if(node->IsLeaf()) return;
    Matrix<real_t>& M = mat_L2L;
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL) {
        FMM_Node* child = node->child[octant];
        std::vector<size_t>& perm_in = perm_r[octant].perm;
        std::vector<size_t>& perm_out = perm_c[octant].perm;
        Matrix<real_t> buffer_in(1, NSURF);
        Matrix<real_t> buffer_out(1, NSURF);
        for(int k=0; k<NSURF; k++) {
          buffer_in[0][k] = node->dnward_equiv[perm_in[k]]; // input perm
        }
        Matrix<real_t>::GEMM(buffer_out, buffer_in, M);
        for(int k=0; k<NSURF; k++)
          child->dnward_equiv[k] += buffer_out[0][perm_out[k]];
      }
    }
    for(int octant=0; octant<8; octant++) {
      if(node->child[octant] != NULL)
        #pragma omp task untied
        L2L(node->child[octant]);
    }
    #pragma omp taskwait
  }

  void L2P() {
    #pragma omp parallel for
    for(int i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      int level = leaf->depth;
      real_t scal = pow(0.5, level);
      // check surface potential -> equivalent surface charge
      Matrix<real_t> check(1, NSURF, &(leaf->dnward_equiv[0]));  // check surface potential
      Matrix<real_t> buffer(1, NSURF);
      Matrix<real_t>::GEMM(buffer, check, L2L_V);
      Matrix<real_t> equiv(1, NSURF);  // equivalent surface charge
      Matrix<real_t>::GEMM(equiv, buffer, L2L_U);
      for(int k=0; k<NSURF; k++)
        leaf->dnward_equiv[k] = scal * equiv[0][k];
      // equivalent surface charge -> target potential
      std::vector<real_t> equivCoord(NSURF*3);
      for(int k=0; k<NSURF; k++) {
        equivCoord[3*k+0] = dnwd_equiv_surf[level][3*k+0] + leaf->coord[0];
        equivCoord[3*k+1] = dnwd_equiv_surf[level][3*k+1] + leaf->coord[1];
        equivCoord[3*k+2] = dnwd_equiv_surf[level][3*k+2] + leaf->coord[2];
      }
      gradientP2P(equivCoord, leaf->dnward_equiv, leaf->pt_coord, leaf->pt_trg);
    }
  }

  void P2L() {
    std::vector<FMM_Node*>& targets = allnodes;
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      if (target->IsLeaf() && target->numBodies<=NSURF)
        continue;
      std::vector<FMM_Node*>& sources = target->interac_list[P2L_Type];
      for(int j=0; j<sources.size(); j++) {
        FMM_Node* source = sources[j];
        if (source != NULL) {
          std::vector<real_t> targetCheckCoord(NSURF*3);
          int level = target->depth;
          // target cell's check coord = relative check coord + cell's origin
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

  void M2P() {
    std::vector<FMM_Node*>& targets = leafs;  // leafs
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      std::vector<FMM_Node*>& sources = target->interac_list[M2P_Type];
      for(int j=0; j<sources.size(); j++) {
        FMM_Node* source = sources[j];
        if (source != NULL) {
          if (source->IsLeaf() && source->numBodies<=NSURF)
            continue;
          std::vector<real_t> sourceEquivCoord(NSURF*3);
          int level = source->depth;
          // source cell's equiv coord = relative equiv coord + cell's origin
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

  void P2P() {
    std::vector<FMM_Node*>& targets = leafs;   // leafs, assume sources == targets
    std::vector<Mat_Type> types = {P2P0_Type, P2P1_Type, P2P2_Type, P2L_Type, M2P_Type};
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      for(int k=0; k<types.size(); k++) {
        Mat_Type type = types[k];
        std::vector<FMM_Node*>& sources = target->interac_list[type];
        if (type == P2L_Type)
          if (target->numBodies > NSURF) {
            continue;
          }
        for(int j=0; j<sources.size(); j++) {
          FMM_Node* source = sources[j];
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

  void M2LListHadamard(std::vector<size_t>& interac_dsp, std::vector<size_t>& interac_vec,
                       AlignedVec& fft_in, AlignedVec& fft_out) {
    AlignedVec zero_vec0(FFTSIZE, 0.);
    AlignedVec zero_vec1(FFTSIZE, 0.);

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
      for(size_t k=0; k<N3_; k++) {
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
#if 0
            if (j+2 < interac_cnt) {
              _mm_prefetch(((char *)(IN[j+2] + k*NCHILD*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+2] + k*NCHILD*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+3] + k*NCHILD*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(IN[j+3] + k*NCHILD*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+2] + k*NCHILD*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+2] + k*NCHILD*2) + 64), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+3] + k*NCHILD*2)), _MM_HINT_T0);
              _mm_prefetch(((char *)(OUT[j+3] + k*NCHILD*2) + 64), _MM_HINT_T0);
            }
#endif
            matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
          }
        }
      }
    }
    Profile::Add_FLOP(8*8*8*(interac_vec.size()/2)*N3_);
  }

  void FFT_UpEquiv(std::vector<size_t>& fft_vec, std::vector<real_t>& fft_scal,
                   std::vector<real_t>& input_data, AlignedVec& fft_in) {
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    std::vector<real_t> surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER-1-surf[i*3]+0.5))
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+1]+0.5)) * N1
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+2]+0.5)) * N2;
    }

    AlignedVec fftw_in(N3 * NCHILD);
    AlignedVec fftw_out(FFTSIZE);
    fft_plan m2l_list_fftplan = fft_plan_many_dft_r2c(3, FFTDIM, NCHILD, 
                                (real_t*)&fftw_in[0], NULL, 1, N3,
                                (fft_complex*)(&fftw_out[0]), NULL, 1, N3_,
                                FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<fft_vec.size(); node_idx++) {
      std::vector<real_t> buffer(FFTSIZE, 0);
      real_t* upward_equiv = &input_data[fft_vec[node_idx]];  // offset ptr of node's 8 child's upward_equiv in allUpwardEquiv, size=8*NSURF
      // upward_equiv_fft (input of r2c) here should have a size of N3*NCHILD
      // the node_idx's chunk of fft_out has a size of 2*N3_*NCHILD
      // since it's larger than what we need,  we can use fft_out as fftw_in buffer here
      real_t* upward_equiv_fft = &fft_in[FFTSIZE*node_idx]; // offset ptr of node_idx in fft_in vector, size=FFTSIZE
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<(int)NCHILD; j0++)
          upward_equiv_fft[idx+j0*N3] = upward_equiv[j0*NSURF+k] * fft_scal[node_idx];
      }
      fft_execute_dft_r2c(m2l_list_fftplan, upward_equiv_fft, (fft_complex*)&buffer[0]);
      for(size_t j=0; j<N3_; j++) {
        for(size_t k=0; k<NCHILD; k++) {
          upward_equiv_fft[2*(NCHILD*j+k)+0] = buffer[2*(N3_*k+j)+0];
          upward_equiv_fft[2*(NCHILD*j+k)+1] = buffer[2*(N3_*k+j)+1];
        }
      }
    }
    fft_destroy_plan(m2l_list_fftplan);
  }

  void FFT_Check2Equiv(std::vector<size_t>& ifft_vec, std::vector<real_t>& ifft_scal,
                       AlignedVec& fft_out, std::vector<real_t>& output_data) {
    std::vector<size_t> map(NSURF);
    real_t c[3]= {0, 0, 0};
    std::vector<real_t> surf = surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0);
    for(size_t i=0; i<map.size(); i++) {
      map[i] = ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3]))
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+1])) * N1
             + ((size_t)(MULTIPOLE_ORDER*2-0.5-surf[i*3+2])) * N2;
    }

    AlignedVec fftw_in(FFTSIZE);
    AlignedVec fftw_out(N3 * NCHILD);
    fft_plan m2l_list_ifftplan = fft_plan_many_dft_c2r(3, FFTDIM, NCHILD,
                                 (fft_complex*)&fftw_in[0], NULL, 1, N3_,
                                 (real_t*)(&fftw_out[0]), NULL, 1, N3,
                                 FFTW_ESTIMATE);
    #pragma omp parallel for
    for(size_t node_idx=0; node_idx<ifft_vec.size(); node_idx++) {
      std::vector<real_t> buffer0(FFTSIZE, 0);
      std::vector<real_t> buffer1(FFTSIZE, 0);
      real_t* dnward_check_fft = &fft_out[FFTSIZE*node_idx];  // offset ptr for node_idx in fft_out vector, size=FFTSIZE
      real_t* dnward_equiv = &output_data[ifft_vec[node_idx]];  // offset ptr for node_idx's child's dnward_equiv in allDnwardEquiv, size=numChilds * NSURF
      for(size_t j=0; j<N3_; j++)
        for(size_t k=0; k<NCHILD; k++) {
          buffer0[2*(N3_*k+j)+0] = dnward_check_fft[2*(NCHILD*j+k)+0];
          buffer0[2*(N3_*k+j)+1] = dnward_check_fft[2*(NCHILD*j+k)+1];
        }
      fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
      for(size_t k=0; k<NSURF; k++) {
        size_t idx = map[k];
        for(int j0=0; j0<NCHILD; j0++)
          dnward_equiv[NSURF*j0+k]+=buffer1[idx+j0*N3]*ifft_scal[node_idx];
      }
    }
    fft_destroy_plan(m2l_list_ifftplan);
  }

  void M2L(M2LData& M2Ldata) {
    size_t numNodes = allnodes.size();
    std::vector<real_t> allUpwardEquiv(numNodes*NSURF);
    std::vector<real_t> allDnwardEquiv(numNodes*NSURF);
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        allUpwardEquiv[i*NSURF+j] = allnodes[i]->upward_equiv[j];
        allDnwardEquiv[i*NSURF+j] = allnodes[i]->dnward_equiv[j];
      }
    }
    AlignedVec fft_in(M2Ldata.fft_vec.size()*FFTSIZE, 0.);
    AlignedVec fft_out(M2Ldata.ifft_vec.size()*FFTSIZE, 0.);

    FFT_UpEquiv(M2Ldata.fft_vec, M2Ldata.fft_scl, allUpwardEquiv, fft_in);
    M2LListHadamard(M2Ldata.interac_dsp, M2Ldata.interac_vec, fft_in, fft_out);
    FFT_Check2Equiv(M2Ldata.ifft_vec, M2Ldata.ifft_scl, fft_out, allDnwardEquiv);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        allnodes[i]->upward_equiv[j] = allUpwardEquiv[i*NSURF+j];
        allnodes[i]->dnward_equiv[j] = allDnwardEquiv[i*NSURF+j];
      }
    }
  }
}//end namespace

#endif
