#ifndef pvfmm_h
#define pvfmm_h
#include "align.h"
#include "vector.hpp"
#include "matrix.hpp"
#include "vec.h"
namespace pvfmm{
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

  typedef vec<3,int> ivec3;                            //!< Vector of 3 int types
  //! SIMD vector types for AVX512, AVX, and SSE
  const int NSIMD = SIMD_BYTES / int(sizeof(real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
  typedef vec<NSIMD,real_t> simdvec;                   //!< SIMD vector type
  typedef AlignedAllocator<real_t, MEM_ALIGN> AlignAllocator;
  typedef std::vector<real_t> RealVec;

  fft_plan vprecomp_fftplan;
  bool vprecomp_fft_flag;
  fft_plan vlist_fftplan;
  bool vlist_fft_flag;
  fft_plan vlist_ifftplan;
  bool vlist_ifft_flag;
  const int SrcCoord = 1, SrcValue = 2, TrgCoord = 3, TrgValue = 4,
            UpwardEquivCoord = 5, UpwardCheckCoord=6, UpwardEquivValue = 7,
            DnwardEquivCoord = 8, DnwardCheckCoord=9, DnwardEquivValue = 10;

  typedef enum{
    M2M_V_Type= 0,
    M2M_U_Type= 1,
    L2L_V_Type= 2,
    L2L_U_Type= 3,
    M2M_Type  = 4,
    L2L_Type  = 5,
    V_Type    = 6,
    V1_Type   = 7,
    P2M_Type  = 8,
    L2P_Type  = 9,
    U0_Type   = 10,
    U1_Type   = 11,
    U2_Type   = 12,
    M2P_Type  = 13,
    P2L_Type    = 14,
    Type_Count= 15
  } Mat_Type;
  const int PrecomputationType = 8;   // first 8 types need precomputation

  typedef enum{
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

  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;    // displacement of cost
    Vector<real_t> scal[4*MAX_DEPTH];
    Matrix<real_t> M[4];   // M is not empty for P2M, L2P, empty for other lists
  };

  struct VListData {
    size_t buff_size;
    size_t m;
    size_t n_blk0;
    std::vector<real_t*> precomp_mat;
    std::vector<std::vector<size_t> > fft_vec;
    std::vector<std::vector<size_t> > ifft_vec;
    std::vector<std::vector<real_t> > fft_scl;
    std::vector<std::vector<real_t> > ifft_scl;
    std::vector<std::vector<size_t> > interac_vec;
    std::vector<std::vector<size_t> > interac_dsp;
  };

  std::vector<Vector<real_t> > upwd_check_surf;
  std::vector<Vector<real_t> > upwd_equiv_surf;
  std::vector<Vector<real_t> > dnwd_check_surf;
  std::vector<Vector<real_t> > dnwd_equiv_surf;

  std::vector<char> M2M_precomp_lst;
  std::vector<char> L2L_precomp_lst;
  std::vector<Matrix<real_t> > node_data_buff;
  Vector<char> dev_buffer;
}
#endif //_PVFMM_COMMON_HPP_
