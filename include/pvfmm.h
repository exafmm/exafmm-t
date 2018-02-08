#ifndef pvfmm_h
#define pvfmm_h
#include "vector.hpp"

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

  std::vector<Vector<real_t> > upwd_check_surf;
  std::vector<Vector<real_t> > upwd_equiv_surf;
  std::vector<Vector<real_t> > dnwd_check_surf;
  std::vector<Vector<real_t> > dnwd_equiv_surf;
}
#endif //_PVFMM_COMMON_HPP_
