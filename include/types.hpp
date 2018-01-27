#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

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

#endif //_PVFMM_COMMON_HPP_
