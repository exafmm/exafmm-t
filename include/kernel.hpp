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
      simdvec tx(&trg_coord[0*trg_cnt+t], (int)sizeof(real_t));
      simdvec ty(&trg_coord[1*trg_cnt+t], (int)sizeof(real_t));
      simdvec tz(&trg_coord[2*trg_cnt+t], (int)sizeof(real_t));
      simdvec tv(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[0*src_cnt+s]);
        sx = sx - tx;
        simdvec sy(src_coord[1*src_cnt+s]);
        sy = sy - ty;
        simdvec sz(src_coord[2*src_cnt+s]);
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
      for(int k=0; k<NSIMD && t+k<trg_cnt; k++)
        trg_value[t+k] = tv[k];
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
      simdvec tx(&trg_coord[0*trg_cnt+t], (int)sizeof(real_t));
      simdvec ty(&trg_coord[1*trg_cnt+t], (int)sizeof(real_t));
      simdvec tz(&trg_coord[2*trg_cnt+t], (int)sizeof(real_t));
      simdvec tv0(zero);
      simdvec tv1(zero);
      simdvec tv2(zero);
      simdvec tv3(zero);
      for(int s=0; s<src_cnt; s++) {
        simdvec sx(src_coord[0*src_cnt+s]);
        sx = tx - sx;
        simdvec sy(src_coord[1*src_cnt+s]);
        sy = ty - sy;
        simdvec sz(src_coord[2*src_cnt+s]);
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
        trg_value[0+4*(t+k)] = tv0[k];
        trg_value[1+4*(t+k)] = tv1[k];
        trg_value[2+4*(t+k)] = tv2[k];
        trg_value[3+4*(t+k)] = tv3[k];
      }
    }
    //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*27);
  }

  //! Wrap around the above P2P functions with matrix interface to provide array interface
  //! Evaluate potential / gradient based on the argument grad
  // r_src & r_trg coordinate array: [x1, y1, z1, x2, y2, z2, ...]
  void laplaceP2P(real_t* r_src, int src_cnt, real_t* v_src, real_t* r_trg, int trg_cnt,
                  real_t* v_trg, bool grad=false) {
    int trg_dim = (grad) ? 4 : 1;
    RealVec src_coord(src_cnt * 3);
    RealVec src_value(src_cnt);
    RealVec trg_coord(trg_cnt * 3);
    RealVec trg_value(trg_cnt * trg_dim, 0.);
    for(size_t i=0; i<src_cnt ; i++) {
      for(size_t j=0; j<3; j++)
        src_coord[i+j*src_cnt] = r_src[i*3+j];
    }
    for(int i=0; i<src_cnt ; i++)
      src_value[i]=v_src[i];
    for(int i=0; i<trg_cnt ; i++) {
      for(size_t j=0; j<3; j++)
        trg_coord[i+j*trg_cnt] = r_trg[i*3+j];
    }
    if (grad) gradientP2P(src_coord, src_value, trg_coord, trg_value);
    else potentialP2P(src_coord, src_value, trg_coord, trg_value);
    for(size_t i=0; i<trg_cnt ; i++) {
      for(size_t j=0; j<trg_dim; j++)
        v_trg[i*trg_dim+j]+=trg_value[i*trg_dim+j];
    }
  }

  //! Laplace potential P2P with array interface
  void potentialP2P(real_t* r_src, int src_cnt, real_t* v_src, real_t* r_trg, int trg_cnt,
                    real_t* v_trg) {
    laplaceP2P(r_src, src_cnt, v_src,  r_trg, trg_cnt, v_trg, false);
  }
  //! Laplace gradient P2P with array interface
  void gradientP2P(real_t* r_src, int src_cnt, real_t* v_src,  real_t* r_trg, int trg_cnt,
                   real_t* v_trg) {
    laplaceP2P(r_src, src_cnt, v_src, r_trg, trg_cnt, v_trg, true);
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  // For Laplace: ker_dim[0] = 1, j = 0; Force a unit charge (q=1)
  // r_src layout: [x1, y1, z1, x2, y2, z2, ...]
  // k_out layout (potential): [p11, p12, p13, ..., p21, p22, ...]  (1st digit: src_idx; 2nd: trg_idx)
  // k_out layout (gradient) : [Fx11, Fy11, Fz11, Fx12, Fy12, Fz13, ... Fx1n, Fy1n, Fz1n, ...
  //                            Fx21, Fy21, Fz21, Fx22, Fy22, Fz22, ... Fx2n, Fy2n, Fz2n, ...
  //                            ...]
  void BuildMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) {
    memset(k_out, 0, src_cnt*SRC_DIM*trg_cnt*POT_DIM*sizeof(real_t));
    for(int i=0; i<src_cnt; i++) {
      for(int j=0; j<SRC_DIM; j++) {
        std::vector<real_t> v_src(SRC_DIM, 0);
        v_src[j]=1.0;
        // do P2P: i-th source
        potentialP2P(&r_src[i*3], 1, &v_src[0], r_trg, trg_cnt,
                  &k_out[(i*SRC_DIM+j)*trg_cnt*POT_DIM]);
      }
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
      potentialP2P(&(leaf->pt_coord[0]), leaf->pt_cnt[0], &(leaf->pt_src[0]),
                               &checkCoord[0], NSURF, &
                               (leaf->upward_equiv[0]));  // save check potentials in upward_equiv temporarily check surface potential -> equivalent surface charge
      Matrix<real_t> check(1, NSURF, &(leaf->upward_equiv[0]), true);  // check surface potential
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
      Matrix<real_t> check(1, NSURF, &(leaf->dnward_equiv[0]), true);  // check surface potential
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
      gradientP2P(&equivCoord[0], NSURF, &(leaf->dnward_equiv[0]),
                 &(leaf->pt_coord[0]), leaf->pt_cnt[1], &(leaf->pt_trg[0]));
    }
  }

  void P2L() {
    std::vector<FMM_Node*>& targets = allnodes;
    #pragma omp parallel for
    for(int i=0; i<targets.size(); i++) {
      FMM_Node* target = targets[i];
      if (target->IsLeaf() && target->pt_cnt[1]<=NSURF)
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
          potentialP2P(&(source->pt_coord[0]), source->pt_cnt[0], &(source->pt_src[0]),
                       &targetCheckCoord[0], NSURF, &(target->dnward_equiv[0]));
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
          if (source->IsLeaf() && source->pt_cnt[0]<=NSURF)
            continue;
          std::vector<real_t> sourceEquivCoord(NSURF*3);
          int level = source->depth;
          // source cell's equiv coord = relative equiv coord + cell's origin
          for(int k=0; k<NSURF; k++) {
            sourceEquivCoord[3*k+0] = upwd_equiv_surf[level][3*k+0] + source->coord[0];
            sourceEquivCoord[3*k+1] = upwd_equiv_surf[level][3*k+1] + source->coord[1];
            sourceEquivCoord[3*k+2] = upwd_equiv_surf[level][3*k+2] + source->coord[2];
          }
          gradientP2P(&sourceEquivCoord[0], NSURF, &(source->upward_equiv[0]),
                       &(target->pt_coord[0]), target->pt_cnt[1], &(target->pt_trg[0]));
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
          if (target->pt_cnt[1] > NSURF)
            continue;
        for(int j=0; j<sources.size(); j++) {
          FMM_Node* source = sources[j];
          if (source != NULL) {
            if (type == M2P_Type)
              if (source->pt_cnt[0] > NSURF)
                continue;
            gradientP2P(&(source->pt_coord[0]), source->pt_cnt[0], &(source->pt_src[0]),
                         &(target->pt_coord[0]), target->pt_cnt[1], &(target->pt_trg[0]));
          }
        }
      }
    }
  }

  void M2LListHadamard(size_t M_dim, std::vector<size_t>& interac_dsp,
                       std::vector<size_t>& interac_vec,
                       std::vector<real_t*>& precomp_mat, Matrix<real_t>& fft_in, Matrix<real_t>& fft_out) {
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =M_dim*chld_cnt*2;
    size_t fftsize_out=M_dim*chld_cnt*2;
    int err;
    real_t * zero_vec0, * zero_vec1;
    err = posix_memalign((void**)&zero_vec0, MEM_ALIGN, fftsize_in *sizeof(real_t));
    err = posix_memalign((void**)&zero_vec1, MEM_ALIGN, fftsize_out*sizeof(real_t));
    size_t n_out=fft_out.dim[0] * fft_out.dim[1]/fftsize_out;
    fft_out.SetZero();

    size_t mat_cnt=precomp_mat.size();
    size_t blk1_cnt=interac_dsp.size()/mat_cnt;
    int BLOCK_SIZE = CACHE_SIZE * 4 / sizeof(real_t);
    real_t **IN_, **OUT_;
    err = posix_memalign((void**)&IN_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    err = posix_memalign((void**)&OUT_, MEM_ALIGN, BLOCK_SIZE*blk1_cnt*mat_cnt*sizeof(real_t*));
    #pragma omp parallel for
    for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++) {
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      for(size_t j=0; j<interac_cnt; j++) {
        IN_ [BLOCK_SIZE*interac_blk1 +j]=&fft_in[0][interac_vec[(interac_dsp0+j)*2+0]];
        OUT_[BLOCK_SIZE*interac_blk1 +j]=&fft_out[0][interac_vec[(interac_dsp0+j)*2+1]];
      }
      IN_ [BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
      OUT_[BLOCK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
    }
    int omp_p=omp_get_max_threads();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++) {
      size_t a=( pid   *M_dim)/omp_p;
      size_t b=((pid+1)*M_dim)/omp_p;
      for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
        for(size_t        k=a;        k<       b;       k++)
          for(size_t mat_indx=0; mat_indx< mat_cnt; mat_indx++) {
            size_t interac_blk1 = blk1*mat_cnt+mat_indx;
            size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
            size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
            size_t interac_cnt  = interac_dsp1-interac_dsp0;
            real_t** IN = IN_ + BLOCK_SIZE*interac_blk1;
            real_t** OUT= OUT_+ BLOCK_SIZE*interac_blk1;
            real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2;
            for(size_t j=0; j<interac_cnt; j+=2) {
              real_t* M_   = M;
              real_t* IN0  = IN [j+0] + k*chld_cnt*2;
              real_t* IN1  = IN [j+1] + k*chld_cnt*2;
              real_t* OUT0 = OUT[j+0] + k*chld_cnt*2;
              real_t* OUT1 = OUT[j+1] + k*chld_cnt*2;
#ifdef __SSE__
              if (j+2 < interac_cnt) {
                _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
                _mm_prefetch(((char *)(IN[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
                _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
                _mm_prefetch(((char *)(IN[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
                _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2)), _MM_HINT_T0);
                _mm_prefetch(((char *)(OUT[j+2] + k*chld_cnt*2) + 64), _MM_HINT_T0);
                _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2)), _MM_HINT_T0);
                _mm_prefetch(((char *)(OUT[j+3] + k*chld_cnt*2) + 64), _MM_HINT_T0);
              }
#endif
              matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
            }
          }
    }
    Profile::Add_FLOP(8*8*8*(interac_vec.size()/2)*M_dim);
    free(IN_ );
    free(OUT_);
    free(zero_vec0);
    free(zero_vec1);
  }

  void FFT_UpEquiv(size_t m, std::vector<size_t>& fft_vec, std::vector<real_t>& fft_scal,
                   std::vector<real_t>& input_data, Matrix<real_t>& output_data) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_in =2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    static std::vector<size_t> map;
    size_t n_old = map.size();
    if(n_old!=NSURF) {
      real_t c[3]= {0, 0, 0};
      std::vector<real_t> surf = surface(m, c, (real_t)(m-1), 0);
      map.resize(surf.size()/3);
      for(size_t i=0; i<map.size(); i++)
        map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(
                 m-1-surf[i*3+2]+0.5))*n2;
    }

    int err, nnn[3]= {(int)n1, (int)n1, (int)n1};
    real_t *fftw_in, *fftw_out;
    err = posix_memalign((void**)&fftw_in,  MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
    err = posix_memalign((void**)&fftw_out, MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
    fft_plan m2l_list_fftplan = fft_plan_many_dft_r2c(3, nnn, chld_cnt,
                                (real_t*)fftw_in, NULL, 1, n3,
                                (fft_complex*)(fftw_out), NULL, 1, n3_,
                                FFTW_ESTIMATE);
    free(fftw_in );
    free(fftw_out);

    size_t n_in = fft_vec.size();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++) {
      size_t node_start=(n_in*(pid  ))/omp_p;
      size_t node_end  =(n_in*(pid+1))/omp_p;
      std::vector<real_t> buffer(fftsize_in, 0);
      for(size_t node_idx=node_start; node_idx<node_end; node_idx++) {
        // upward_equiv.size is numChilds * NSURF
        real_t* upward_equiv = &input_data[fft_vec[node_idx]];  // offset ptr for node_idx's child's upward_equiv in allUpwardEquiv
        //real_t* upward_equiv_fft = &output_data[0][fftsize_in*node_idx];  // offset ptr for node_idx in fft_in vector
        Matrix<real_t> upward_equiv_fft(1, fftsize_in, &output_data[0][fftsize_in*node_idx], false);
        upward_equiv_fft.SetZero();
        for(size_t k=0; k<NSURF; k++) {
          size_t idx=map[k];
          int j1=0;
          for(int j0=0; j0<(int)chld_cnt; j0++)
            upward_equiv_fft[0][idx+j0*n3] = upward_equiv[j0*NSURF+k] * fft_scal[node_idx];
        }
        fft_execute_dft_r2c(m2l_list_fftplan, (real_t*)&upward_equiv_fft[0][0], (fft_complex*)&buffer[0]);
        for(size_t j=0; j<n3_; j++) {
          for(size_t k=0; k<chld_cnt; k++) {
            upward_equiv_fft[0][2*(chld_cnt*j+k)+0]=buffer[2*(n3_*k+j)+0];
            upward_equiv_fft[0][2*(chld_cnt*j+k)+1]=buffer[2*(n3_*k+j)+1];
          }
        }
      }
    }
    fft_destroy_plan(m2l_list_fftplan);
  }

  void FFT_Check2Equiv(size_t m, std::vector<size_t>& ifft_vec, std::vector<real_t>& ifft_scal,
                       Matrix<real_t>& input_data, std::vector<real_t>& output_data) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<3;
    size_t fftsize_out=2*n3_*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static std::vector<size_t> map;
    size_t n_old = map.size();
    if(n_old!=n) {
      real_t c[3]= {0, 0, 0};
      std::vector<real_t> surf = surface(m, c, (real_t)(m-1), 0);
      map.resize(surf.size()/3);
      for(size_t i=0; i<map.size(); i++)
        map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(
                 m*2-0.5-surf[i*3+2]))*n2;
    }

    int err, nnn[3]= {(int)n1, (int)n1, (int)n1};
    real_t *fftw_in, *fftw_out;
    err = posix_memalign((void**)&fftw_in,  MEM_ALIGN, 2*n3_*chld_cnt*sizeof(real_t));
    err = posix_memalign((void**)&fftw_out, MEM_ALIGN,   n3 *chld_cnt*sizeof(real_t));
    fft_plan m2l_list_ifftplan = fft_plan_many_dft_c2r(3, nnn, chld_cnt,
                                 (fft_complex*)fftw_in, NULL, 1, n3_,
                                 (real_t*)(fftw_out), NULL, 1, n3,
                                 FFTW_ESTIMATE);
    free(fftw_in);
    free(fftw_out);

    size_t n_out=ifft_vec.size();
    #pragma omp parallel for
    for(int pid=0; pid<omp_p; pid++) {
      size_t node_start=(n_out*(pid  ))/omp_p;
      size_t node_end  =(n_out*(pid+1))/omp_p;
      std::vector<real_t> buffer0(fftsize_out, 0);
      std::vector<real_t> buffer1(fftsize_out, 0);
      for(size_t node_idx=node_start; node_idx<node_end; node_idx++) {
        real_t* dnward_check_fft = &input_data[0][fftsize_out*node_idx];  // offset ptr for node_idx in fft_out vector
        // dnward_equiv.size is numChilds * NSURF
        real_t* dnward_equiv = &output_data[ifft_vec[node_idx]];  // offset ptr for node_idx's child's dnward_equiv in allDnwardEquiv
        for(size_t j=0; j<n3_; j++)
          for(size_t k=0; k<chld_cnt; k++) {
            buffer0[2*(n3_*k+j)+0]=dnward_check_fft[2*(chld_cnt*j+k)+0];
            buffer0[2*(n3_*k+j)+1]=dnward_check_fft[2*(chld_cnt*j+k)+1];
          }
        fft_execute_dft_c2r(m2l_list_ifftplan, (fft_complex*)&buffer0[0], (real_t*)&buffer1[0]);
        for(size_t k=0; k<n; k++) {
          size_t idx=map[k];
          for(int j0=0; j0<(int)chld_cnt; j0++)
            dnward_equiv[n*j0+k]+=buffer1[idx+j0*n3]*ifft_scal[node_idx];
        }
      }
    }
    fft_destroy_plan(m2l_list_ifftplan);
  }

  void M2L(M2LData& M2Ldata) {
    size_t numNodes = allnodes.size();
    #pragma omp parallel for collapse(2)
    for(int i=0; i<numNodes; i++) {
      for(int j=0; j<NSURF; j++) {
        allUpwardEquiv[i*NSURF+j] = allnodes[i]->upward_equiv[j];
        allDnwardEquiv[i*NSURF+j] = allnodes[i]->dnward_equiv[j];
      }
    }
    size_t buffersize = 1024*1024*1024;
    Matrix<real_t> buffer(1, buffersize);
    real_t* buff = buffer.data_ptr;
    size_t n_blk0 = M2Ldata.n_blk0;
    size_t m = MULTIPOLE_ORDER;
    size_t n1 = m * 2;
    size_t n2 = n1 * n1;
    size_t n3_ = n2 * (n1 / 2 + 1);
    size_t chld_cnt = 8;
    size_t fftsize = 2 * n3_ * chld_cnt;
    size_t M_dim = n3_;
    std::vector<real_t*> precomp_mat = M2Ldata.precomp_mat;
    std::vector<std::vector<size_t> >&  fft_vec = M2Ldata.fft_vec;
    std::vector<std::vector<size_t> >& ifft_vec = M2Ldata.ifft_vec;
    std::vector<std::vector<real_t> >&  fft_scl = M2Ldata.fft_scl;
    std::vector<std::vector<real_t> >& ifft_scl = M2Ldata.ifft_scl;
    std::vector<std::vector<size_t> >& interac_vec = M2Ldata.interac_vec;
    std::vector<std::vector<size_t> >& interac_dsp = M2Ldata.interac_dsp;
    for(size_t blk0=0; blk0<n_blk0; blk0++) {
      size_t n_in = fft_vec[blk0].size();  // num of nodes_in in this block
      size_t n_out=ifft_vec[blk0].size();  // num of nodes_out in this block
      size_t  input_dim=n_in *fftsize;
      size_t output_dim=n_out*fftsize;
      //std::vector<real_t> fft_in(n_in * fftsize, 0);
      //AlignedVec fft_out(n_out * fftsize, 0);  // fft_out must be aligned
      Matrix<real_t> fft_in(1, input_dim, (real_t*)buff, false);
      Matrix<real_t> fft_out(1, output_dim, (real_t*)(buff+input_dim*sizeof(real_t)), false);
      Profile::Tic("FFT_UpEquiv", false, 5);
      FFT_UpEquiv(m, fft_vec[blk0],  fft_scl[blk0], allUpwardEquiv, fft_in);
      Profile::Toc();
      Profile::Tic("M2LHadamard", false, 5);
      M2LListHadamard(M_dim, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
      Profile::Toc();
      Profile::Tic("FFT_Check2Equiv", false, 5);
      FFT_Check2Equiv(m, ifft_vec[blk0], ifft_scl[blk0], fft_out, allDnwardEquiv);
      Profile::Toc();
    }

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
