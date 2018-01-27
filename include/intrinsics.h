#ifndef intrinsics_h
#define intrinsics_h
#include "pvfmm.h"

namespace pvfmm {
#ifdef __AVX__
  inline __m256 zero_intrin(const float){
    return _mm256_setzero_ps();
  }

  inline __m256d zero_intrin(const double){
    return _mm256_setzero_pd();
  }

  inline __m256 set_intrin(const float& a){
    return _mm256_set_ps(a,a,a,a,a,a,a,a);
  }

  inline __m256d set_intrin(const double& a){
    return _mm256_set_pd(a,a,a,a);
  }

  inline __m256 load_intrin(const float* a){
    return _mm256_load_ps(a);
  }

  inline __m256d load_intrin(const double* a){
    return _mm256_load_pd(a);
  }

  inline void store_intrin(float* a, const __m256& b){
    return _mm256_store_ps(a,b);
  }

  inline void store_intrin(double* a, const __m256d& b){
    return _mm256_store_pd(a,b);
  }

  inline __m256 mul_intrin(const __m256& a, const __m256& b){
    return _mm256_mul_ps(a,b);
  }

  inline __m256d mul_intrin(const __m256d& a, const __m256d& b){
    return _mm256_mul_pd(a,b);
  }

  inline __m256 add_intrin(const __m256& a, const __m256& b){
    return _mm256_add_ps(a,b);
  }

  inline __m256d add_intrin(const __m256d& a, const __m256d& b){
    return _mm256_add_pd(a,b);
  }

  inline __m256 sub_intrin(const __m256& a, const __m256& b){
    return _mm256_sub_ps(a,b);
  }

  inline __m256d sub_intrin(const __m256d& a, const __m256d& b){
    return _mm256_sub_pd(a,b);
  }

  inline __m128 rsqrt_approx_intrin(const __m128& r2){
    return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
  }

  inline __m256 rsqrt_approx_intrin(const __m256& r2){
    return _mm256_andnot_ps(_mm256_cmp_ps(r2,_mm256_setzero_ps(),_CMP_EQ_OS),_mm256_rsqrt_ps(r2));
  }

  inline __m256d rsqrt_approx_intrin(const __m256d& r2){
    return _mm256_cvtps_pd(rsqrt_approx_intrin(_mm256_cvtpd_ps(r2)));
  }

  inline void rsqrt_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

  inline void rsqrt_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

#else
#ifdef __SSE3__
  inline __m128 zero_intrin(const float){
    return _mm_setzero_ps();
  }

  inline __m128d zero_intrin(const double){
    return _mm_setzero_pd();
  }

  inline __m128 set_intrin(const float& a){
    return _mm_set1_ps(a);
  }

  inline __m128d set_intrin(const double& a){
    return _mm_set1_pd(a);
  }

  inline __m128 load_intrin(const float* a){
    return _mm_load_ps(a);
  }

  inline __m128d load_intrin(const double* a){
    return _mm_load_pd(a);
  }

  inline void store_intrin(float* a, const __m128& b){
    return _mm_store_ps(a,b);
  }

  inline void store_intrin(double* a, const __m128d& b){
    return _mm_store_pd(a,b);
  }

  inline __m128 mul_intrin(const __m128& a, const __m128& b){
    return _mm_mul_ps(a,b);
  }

  inline __m128d mul_intrin(const __m128d& a, const __m128d& b){
    return _mm_mul_pd(a,b);
  }

  inline __m128 add_intrin(const __m128& a, const __m128& b){
    return _mm_add_ps(a,b);
  }

  inline __m128d add_intrin(const __m128d& a, const __m128d& b){
    return _mm_add_pd(a,b);
  }

  inline __m128 sub_intrin(const __m128& a, const __m128& b){
    return _mm_sub_ps(a,b);
  }

  inline __m128d sub_intrin(const __m128d& a, const __m128d& b){
    return _mm_sub_pd(a,b);
  }

  inline __m128 rsqrt_approx_intrin(const __m128& r2){
    return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
  }

  inline __m128d rsqrt_approx_intrin(const __m128d& r2){
    return _mm_cvtps_pd(rsqrt_approx_intrin(_mm_cvtpd_ps(r2)));
  }

  inline void rsqrt_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

  inline void rsqrt_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
    rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  }

#endif //__SSE3__
#endif //__AVX__

  inline vec_t rsqrt_intrin2(vec_t r2){
    vec_t rinv=rsqrt_approx_intrin(r2);
    rsqrt_newton_intrin(rinv,r2,real_t(3));
    rsqrt_newton_intrin(rinv,r2,real_t(12));
    return rinv;
  }

#ifdef __AVX__
inline void matmult_8x8x2(double*& M_, double*& IN0, double*& IN1, double*& OUT0, double*& OUT1){
  __m256d out00,out01,out10,out11;
  __m256d out20,out21,out30,out31;
  double* in0__ = IN0;
  double* in1__ = IN1;
  out00 = _mm256_load_pd(OUT0);
  out01 = _mm256_load_pd(OUT1);
  out10 = _mm256_load_pd(OUT0+4);
  out11 = _mm256_load_pd(OUT1+4);
  out20 = _mm256_load_pd(OUT0+8);
  out21 = _mm256_load_pd(OUT1+8);
  out30 = _mm256_load_pd(OUT0+12);
  out31 = _mm256_load_pd(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m256d m00;
    __m256d ot00;
    __m256d mt0,mtt0;
    __m256d in00,in00_r,in01,in01_r;
    in00 = _mm256_broadcast_pd((const __m128d*)in0__);
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*)in1__);
    in01_r = _mm256_permute_pd(in01,5);
    m00 = _mm256_load_pd(M_);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+4);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+8);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+12);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    in00 = _mm256_broadcast_pd((const __m128d*) (in0__+2));
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*) (in1__+2));
    in01_r = _mm256_permute_pd(in01,5);
    m00 = _mm256_load_pd(M_+16);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+20);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+24);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+28);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm256_store_pd(OUT0,out00);
  _mm256_store_pd(OUT1,out01);
  _mm256_store_pd(OUT0+4,out10);
  _mm256_store_pd(OUT1+4,out11);
  _mm256_store_pd(OUT0+8,out20);
  _mm256_store_pd(OUT1+8,out21);
  _mm256_store_pd(OUT0+12,out30);
  _mm256_store_pd(OUT1+12,out31);
}
#endif

#ifdef __SSE3__
inline void matmult_8x8x2(float*& M_, float*& IN0, float*& IN1, float*& OUT0, float*& OUT1){
  __m128 out00,out01,out10,out11;
  __m128 out20,out21,out30,out31;
  float* in0__ = IN0;
  float* in1__ = IN1;
  out00 = _mm_load_ps(OUT0);
  out01 = _mm_load_ps(OUT1);
  out10 = _mm_load_ps(OUT0+4);
  out11 = _mm_load_ps(OUT1+4);
  out20 = _mm_load_ps(OUT0+8);
  out21 = _mm_load_ps(OUT1+8);
  out30 = _mm_load_ps(OUT0+12);
  out31 = _mm_load_ps(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m128 m00;
    __m128 mt0,mtt0;
    __m128 in00,in00_r,in01,in01_r;
    in00 = _mm_castpd_ps(_mm_load_pd1((const double*)in0__));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*)in1__));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
    m00 = _mm_load_ps(M_);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+4);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01  ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+8);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+12);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,  in00));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
    in00 = _mm_castpd_ps(_mm_load_pd1((const double*) (in0__+2)));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*) (in1__+2)));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
    m00 = _mm_load_ps(M_+16);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+20);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01 ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+24);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+28);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm_store_ps(OUT0,out00);
  _mm_store_ps(OUT1,out01);
  _mm_store_ps(OUT0+4,out10);
  _mm_store_ps(OUT1+4,out11);
  _mm_store_ps(OUT0+8,out20);
  _mm_store_ps(OUT1+8,out21);
  _mm_store_ps(OUT0+12,out30);
  _mm_store_ps(OUT1+12,out31);
}
#endif

}
#endif
