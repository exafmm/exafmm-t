#ifndef pvfmm_h
#define pvfmm_h
#include "align.h"
#include "matrix.hpp"
#include "vec.h"
namespace pvfmm {
#ifndef NULL
#define NULL 0
#endif

#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX_DEPTH 62
#define MEM_ALIGN 64
#define CACHE_SIZE 512

#if FLOAT
typedef float real_t;
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#else
typedef double real_t;
typedef fftw_complex fft_complex;
typedef fftw_plan fft_plan;
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#endif

typedef vec<3, int> ivec3;                           //!< std::vector of 3 int types
//! SIMD vector types for AVX512, AVX, and SSE
const int NSIMD = SIMD_BYTES / int(sizeof(
                                     real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
typedef vec<NSIMD, real_t> simdvec;                  //!< SIMD vector type
typedef AlignedAllocator<real_t, MEM_ALIGN> AlignAllocator;
typedef std::vector<real_t> RealVec;
typedef std::vector<real_t, AlignAllocator> AlignedVec;

fft_plan m2l_precomp_fftplan;
bool m2l_precomp_fft_flag;
fft_plan m2l_list_fftplan;
bool m2l_list_fft_flag;
fft_plan m2l_list_ifftplan;
bool m2l_list_ifft_flag;

typedef enum {
  M2M_V_Type= 0,
  M2M_U_Type= 1,
  L2L_V_Type= 2,
  L2L_U_Type= 3,
  M2M_Type  = 4,
  L2L_Type  = 5,
  M2L_Helper_Type    = 6,
  M2L_Type   = 7,
  P2P0_Type   = 10,
  P2P1_Type   = 11,
  P2P2_Type   = 12,
  M2P_Type  = 13,
  P2L_Type    = 14,
  Type_Count= 15
} Mat_Type;
const int PrecomputationType = 8;   // first 8 types need precomputation

typedef enum {
  Scaling = 0,
  ReflecX = 1,
  ReflecY = 2,
  ReflecZ = 3,
  SwapXY  = 4,
  SwapXZ  = 5,
  R_Perm = 0,
  C_Perm = 6,
  Perm_Count=12
} Perm_Type;

struct InitData {
  int max_depth;
  size_t max_pts;
  std::vector<real_t> coord;
  std::vector<real_t> value;
};

struct M2LData {
  size_t n_blk0;
  std::vector<real_t*> precomp_mat;
  std::vector<std::vector<size_t> > fft_vec;   // source's first child's upward_equiv's displacement
  std::vector<std::vector<size_t> > ifft_vec;  // target's first child's dnward_equiv's displacement
  std::vector<std::vector<real_t> > fft_scl;
  std::vector<std::vector<real_t> > ifft_scl;
  std::vector<std::vector<size_t> > interac_vec;
  std::vector<std::vector<size_t> > interac_dsp;
};

std::vector<std::vector<real_t> > upwd_check_surf;
std::vector<std::vector<real_t> > upwd_equiv_surf;
std::vector<std::vector<real_t> > dnwd_check_surf;
std::vector<std::vector<real_t> > dnwd_equiv_surf;

std::vector<real_t> allUpwardEquiv;
std::vector<real_t> allDnwardEquiv;

class FMM_Node;
std::vector<FMM_Node*> leafs, nonleafs, allnodes;
std::vector<std::vector<Matrix<real_t>>> gPrecompMat;

int LEVEL;     // depth of octree
int MULTIPOLE_ORDER;         // order of multipole expansion
int NSURF;     // number of surface coordinates
Matrix<real_t> BUFFER;
}
#endif //_PVFMM_COMMON_HPP_
