#ifndef pvfmm_h
#define pvfmm_h
#include "vector.hpp"
#include "matrix.hpp"

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
#if defined __AVX__
  typedef __m256 vec_t;
#elif defined __SSE3__
  typedef __m128 vec_t;
#else
  typedef real_t vec_t;
#endif
#else
  typedef double real_t;
#if defined __AVX__
  typedef __m256d vec_t;
#elif defined __SSE3__
  typedef __m128d vec_t;
#else
  typedef real_t vec_t;
#endif
#endif

#if FLOAT
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#else
typedef fftw_complex fft_complex;
typedef fftw_plan fft_plan;
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#endif

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
    P2M_Type  = 4,
    M2M_Type  = 5,
    L2L_Type  = 6,
    L2P_Type  = 7,
    U0_Type   = 8,
    U1_Type   = 9,
    U2_Type   =10,
    V_Type    =11,
    W_Type    =12,
    X_Type    =13,
    V1_Type   =14,
    Type_Count=15
  } Mat_Type;

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

  struct FMM_Data{
    Vector<real_t> upward_equiv;
    Vector<real_t> dnward_equiv;
  };

  struct InitData {
    int max_depth;
    size_t max_pts;
    Vector<real_t> coord;
    Vector<real_t> value;
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
}
#endif //_PVFMM_COMMON_HPP_
