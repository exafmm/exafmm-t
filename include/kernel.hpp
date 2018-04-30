#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_
#include "intrinsics.h"
#include "vec.h"
#include "pvfmm.h"
#include "fmm_node.hpp"
namespace pvfmm {

struct Kernel {
 public:
  typedef void (*Ker_t)(real_t* r_src, int src_cnt, real_t* v_src,
                        real_t* r_trg, int trg_cnt, real_t* k_out);

  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  bool init;
  std::vector<real_t> src_scal;
  std::vector<real_t> trg_scal;
  std::vector<Permutation<real_t>> perm_vec;
  std::vector<Permutation<real_t>> perm_r;
  std::vector<Permutation<real_t>> perm_c;

  Kernel* k_p2m;
  Kernel* k_p2l;
  Kernel* k_p2p;
  Kernel* k_m2m;
  Kernel* k_m2l;
  Kernel* k_m2p;
  Kernel* k_l2l;
  Kernel* k_l2p;

  Kernel(Ker_t poten, const char* name, std::pair<int, int> k_dim) {
    ker_dim[0]=k_dim.first;
    ker_dim[1]=k_dim.second;
    ker_name=std::string(name);
    ker_poten=poten;
    init=false;
    src_scal.resize(ker_dim[0], 0.);
    trg_scal.resize(ker_dim[1], 0.);
    perm_vec.resize(Perm_Count);
    std::fill(perm_vec.begin(), perm_vec.begin()+C_Perm, Permutation<real_t>(ker_dim[0]));
    std::fill(perm_vec.begin()+C_Perm, perm_vec.end(), Permutation<real_t>(ker_dim[1]));
    k_p2m=NULL;
    k_p2l=NULL;
    k_p2p=NULL;
    k_m2m=NULL;
    k_m2l=NULL;
    k_m2p=NULL;
    k_l2l=NULL;
    k_l2p=NULL;
  }

  void Initialize() {
    if(init) return;
    init = true;
    // hardcoded src & trg scal here, since they are constants for Laplace based on the original code
    if (ker_dim[1] == 3) std::fill(trg_scal.begin(), trg_scal.end(), 2.);
    else std::fill(trg_scal.begin(), trg_scal.end(), 1.);
    if(!k_p2m) k_p2m=this;
    if(!k_p2l) k_p2l=this;
    if(!k_p2p) k_p2p=this;
    if(!k_m2m) k_m2m=this;
    if(!k_m2l) k_m2l=this;
    if(!k_m2p) k_m2p=this;
    if(!k_l2l) k_l2l=this;
    if(!k_l2p) k_l2p=this;
    assert(k_p2p->ker_dim[0]==ker_dim[0]);
    assert(k_p2m->ker_dim[0]==k_p2l->ker_dim[0]);
    assert(k_p2m->ker_dim[0]==k_p2p->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2l->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2p->ker_dim[0]);
    assert(k_l2l->ker_dim[0]==k_l2p->ker_dim[0]);
    assert(k_p2p->ker_dim[1]==ker_dim[1]);
    assert(k_p2m->ker_dim[1]==k_m2m->ker_dim[1]);
    assert(k_p2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_m2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_p2p->ker_dim[1]==k_m2p->ker_dim[1]);
    assert(k_p2p->ker_dim[1]==k_l2p->ker_dim[1]);
    k_p2m->Initialize();
    k_p2l->Initialize();
    k_p2p->Initialize();
    k_m2m->Initialize();
    k_m2l->Initialize();
    k_m2p->Initialize();
    k_l2l->Initialize();
    k_l2p->Initialize();
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  // For Laplace: ker_dim[0] = 1, j = 0; Force a unit charge (q=1)
  // r_src layout: [x1, y1, z1, x2, y2, z2, ...]
  // k_out layout (potential): [p11, p12, p13, ..., p21, p22, ...]  (1st digit: src_idx; 2nd: trg_idx)
  // k_out layout (gradient) : [Fx11, Fy11, Fz11, Fx12, Fy12, Fz13, ... Fx1n, Fy1n, Fz1n, ...
  //                            Fx21, Fy21, Fz21, Fx22, Fy22, Fz22, ... Fx2n, Fy2n, Fz2n, ...
  //                            ...]
  void BuildMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) const {
    memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(real_t));
    for(int i=0; i<src_cnt; i++)
      for(int j=0; j<ker_dim[0]; j++) {
        std::vector<real_t> v_src(ker_dim[0], 0);
        v_src[j]=1.0;
        // do P2P: i-th source
        ker_poten(&r_src[i*3], 1, &v_src[0], r_trg, trg_cnt,
                  &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]]);
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
      ker_poten(&(leaf->pt_coord[0]), leaf->pt_cnt[0], &(leaf->pt_src[0]),
                               &checkCoord[0], NSURF, &
                               (leaf->upward_equiv[0]));  // save check potentials in upward_equiv temporarily check surface potential -> equivalent surface charge
      Matrix<real_t> check(1, NSURF, &(leaf->upward_equiv[0]), true);  // check surface potential
      Matrix<real_t> buffer(1, NSURF);
      Matrix<real_t>::GEMM(buffer, check, gPrecompMat[M2M_V_Type][0]);
      Matrix<real_t> equiv(1, NSURF);  // equivalent surface charge
      Matrix<real_t>::GEMM(equiv, buffer, gPrecompMat[M2M_U_Type][0]);
      for(int k=0; k<NSURF; k++)
        leaf->upward_equiv[k] = scal * equiv[0][k];
    }
  }

  void M2M(FMM_Node* node) {
    if(node->IsLeaf()) return;
    Matrix<real_t>& M = gPrecompMat[M2M_Type][7];  // 7 is the class coord, will generalize it later
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
};

template<void (*A)(real_t*, int, real_t*, real_t*, int, real_t*)>
Kernel BuildKernel(const char* name, std::pair<int, int> k_dim,
                   Kernel* k_p2m=NULL, Kernel* k_p2l=NULL, Kernel* k_p2p=NULL,
                   Kernel* k_m2m=NULL,
                   Kernel* k_m2l=NULL, Kernel* k_m2p=NULL, Kernel* k_l2l=NULL,
                   Kernel* k_l2p=NULL) {
  Kernel K(A, name, k_dim);
  K.k_p2m=k_p2m;
  K.k_p2l=k_p2l;
  K.k_p2p=k_p2p;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2p=k_m2p;
  K.k_l2l=k_l2l;
  K.k_l2p=k_l2p;
  return K;
}

void potentialP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  simdvec zero((real_t)0);
  const real_t OOFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
  simdvec oofp(OOFP);
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
      r2 = r2 + sx*sx;
      r2 = r2 + sy*sy;
      r2 = r2 + sz*sz;
      simdvec invR = rsqrt(r2);
      invR &= r2 > zero;
      tv = tv + invR*sv;
    }
    tv = tv * oofp;
    for(int k=0; k<NSIMD && t+k<trg_cnt; k++)
      trg_value[t+k] = tv[k];
  }
  //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*20);
}

void gradientP2P(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value) {
  simdvec zero((real_t)0);
  const real_t OOFP = -1.0/(4*2*2*6*M_PI);
  simdvec oofp(OOFP);
  int src_cnt = src_coord.size() / 3;
  int trg_cnt = trg_coord.size() / 3;
  for(int t=0; t<trg_cnt; t+=NSIMD) {
    simdvec tx(&trg_coord[0*trg_cnt+t], (int)sizeof(real_t));
    simdvec ty(&trg_coord[1*trg_cnt+t], (int)sizeof(real_t));
    simdvec tz(&trg_coord[2*trg_cnt+t], (int)sizeof(real_t));
    simdvec tv0(zero);
    simdvec tv1(zero);
    simdvec tv2(zero);
    for(int s=0; s<src_cnt; s++) {
      simdvec sx(src_coord[0*src_cnt+s]);
      sx = tx - sx;
      simdvec sy(src_coord[1*src_cnt+s]);
      sy = ty - sy;
      simdvec sz(src_coord[2*src_cnt+s]);
      sz = tz - sz;
      simdvec r2(zero);
      r2 = r2 + sx*sx;
      r2 = r2 + sy*sy;
      r2 = r2 + sz*sz;
      simdvec invR = rsqrt(r2);
      invR &= r2 > zero;
      simdvec invR3 = (invR*invR) * invR;
      simdvec sv(src_value[s]);
      sv = invR3 * sv;
      tv0 = tv0 + sv*sx;
      tv1 = tv1 + sv*sy;
      tv2 = tv2 + sv*sz;
    }
    tv0 = tv0 * oofp;
    tv1 = tv1 * oofp;
    tv2 = tv2 * oofp;
    for(int k=0; k<NSIMD && t+k<trg_cnt; k++) {
      trg_value[0+3*(t+k)] = tv0[k];
      trg_value[1+3*(t+k)] = tv1[k];
      trg_value[2+3*(t+k)] = tv2[k];
    }
  }
  //Profile::Add_FLOP((long long)trg_cnt*(long long)src_cnt*27);
}

//! Wrap around the above P2P functions with matrix interface to provide array interface
//! Evaluate potential / gradient based on the argument grad
// r_src & r_trg coordinate array: [x1, y1, z1, x2, y2, z2, ...]
void laplaceP2P(real_t* r_src, int src_cnt, real_t* v_src, real_t* r_trg, int trg_cnt,
                real_t* v_trg, bool grad=false) {
  int TRG_DIM = (grad) ? 3 : 1;
  RealVec src_coord(src_cnt * 3);
  RealVec src_value(src_cnt);
  RealVec trg_coord(trg_cnt * 3);
  RealVec trg_value(trg_cnt * TRG_DIM, 0.);
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
    for(size_t j=0; j<TRG_DIM; j++)
      v_trg[i*TRG_DIM+j]+=trg_value[i*TRG_DIM+j];
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


}//end namespace
#endif //_PVFMM_FMM_KERNEL_HPP_
