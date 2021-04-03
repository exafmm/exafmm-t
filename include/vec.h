#ifndef vec_h
#define vec_h
#include <ostream>

#if __MIC__ | __AVX512F__
const int SIMD_BYTES = 64;
#elif __AVX__
const int SIMD_BYTES = 32;
#elif __SSE__
const int SIMD_BYTES = 16;
#else
#error no SIMD
#endif

#ifndef EXAFMM_RSQRT_APPROX
#define EXAFMM_RSQRT_APPROX 1
#endif

namespace exafmm_t {
#ifndef __CUDACC__
  template<int N, typename T>
  class vec {
  private:
    T data[N];
  public:
    vec(){}
    vec(const T &v) {
      for (int i=0; i<N; i++) data[i] = v;
    }
    vec(const vec &v) {
      for (int i=0; i<N; i++) data[i] = v[i];
    }
    ~vec(){}
    const vec &operator=(const T v) {
      for (int i=0; i<N; i++) data[i] = v;
      return *this;
    }
    const vec &operator+=(const T v) {
      for (int i=0; i<N; i++) data[i] += v;
      return *this;
    }
    const vec &operator-=(const T v) {
      for (int i=0; i<N; i++) data[i] -= v;
      return *this;
    }
    const vec &operator*=(const T v) {
      for (int i=0; i<N; i++) data[i] *= v;
      return *this;
    }
    const vec &operator/=(const T v) {
      for (int i=0; i<N; i++) data[i] /= v;
      return *this;
    }
    const vec &operator>=(const T v) {
      for (int i=0; i<N; i++) data[i] >= v;
      return *this;
    }
    const vec &operator<=(const T v) {
      for (int i=0; i<N; i++) data[i] <= v;
      return *this;
    }
    const vec &operator&=(const T v) {
      for (int i=0; i<N; i++) data[i] &= v;
      return *this;
    }
    const vec &operator|=(const T v) {
      for (int i=0; i<N; i++) data[i] |= v;
      return *this;
    }
    const vec &operator=(const vec & v) {
      for (int i=0; i<N; i++) data[i] = v[i];
      return *this;
    }
    const vec &operator+=(const vec & v) {
      for (int i=0; i<N; i++) data[i] += v[i];
      return *this;
    }
    const vec &operator-=(const vec & v) {
      for (int i=0; i<N; i++) data[i] -= v[i];
      return *this;
    }
    const vec &operator*=(const vec & v) {
      for (int i=0; i<N; i++) data[i] *= v[i];
      return *this;
    }
    const vec &operator/=(const vec & v) {
      for (int i=0; i<N; i++) data[i] /= v[i];
      return *this;
    }
    const vec &operator&=(const vec & v) {
      for (int i=0; i<N; i++) data[i] &= v[i];
      return *this;
    }
    const vec &operator|=(const vec & v) {
      for (int i=0; i<N; i++) data[i] |= v[i];
      return *this;
    }
    vec operator+(const T v) const {
      return vec(*this) += v;
    }
    vec operator-(const T v) const {
      return vec(*this) -= v;
    }
    vec operator*(const T v) const {
      return vec(*this) *= v;
    }
    vec operator/(const T v) const {
      return vec(*this) /= v;
    }
    vec operator>(const T v) const {
      return vec(*this) >= v;
    }
    vec operator<(const T v) const {
      return vec(*this) <= v;
    }
    vec operator&(const T v) const {
      return vec(*this) &= v;
    }
    vec operator|(const T v) const {
      return vec(*this) |= v;
    }
    vec operator+(const vec & v) const {
      return vec(*this) += v;
    }
    vec operator-(const vec & v) const {
      return vec(*this) -= v;
    }
    vec operator*(const vec & v) const {
      return vec(*this) *= v;
    }
    vec operator/(const vec & v) const {
      return vec(*this) /= v;
    }
    bool operator<(const vec & v) {
      bool res = true;
      for (int i=0; i<N; i++) res = res && (data[i] < v[i]);
      return res;
    }
    bool operator>(const vec & v) {
      bool res = true;
      for (int i=0; i<N; i++) res = res && (data[i] > v[i]);
      return res;
    }
    bool operator<=(const vec & v) {
      bool res = true;
      for (int i=0; i<N; i++) res = res && (data[i] <= v[i]);
      return res;
    }
    bool operator>=(const vec & v) {
      bool res = true;
      for (int i=0; i<N; i++) res = res && (data[i] >= v[i]);
      return res;
    }
    bool operator==(const vec & v) {
      bool res = true;
      for (int i=0; i<N; i++) res = res && (data[i] == v[i]);
      return res;
    }
    bool operator!=(const vec & v) {
      return (!(*this==v));
    }
    vec operator&(const vec & v) const {
      return vec(*this) &= v;
    }
    vec operator|(const vec & v) const {
      return vec(*this) |= v;
    }
    vec operator-() const {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = -data[i];
      return temp;
    }
    T &operator[](int i) {
      return data[i];
    }
    const T &operator[](int i) const {
      return data[i];
    }
    operator       T* ()       {return data;}
    operator const T* () const {return data;}
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<N; i++) s << v[i] << ' ';
      return s;
    }
    friend T sum(const vec & v) {
      T temp = 0;
      for (int i=0; i<N; i++) temp += v[i];
      return temp;
    }
    friend T norm(const vec & v) {
      T temp = 0;
      for (int i=0; i<N; i++) temp += v[i] * v[i];
      return temp;
    }
    friend vec min(const vec & v, const vec & w) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] < w[i] ? v[i] : w[i];
      return temp;
    }
    friend vec max(const vec & v, const vec & w) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] > w[i] ? v[i] : w[i];
      return temp;
    }
    friend T min(const vec & v) {
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    friend T max(const vec & v) {
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }
    friend vec sin(const vec & v) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = sin(v[i]);
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = cos(v[i]);
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = exp(v[i]);
      return temp;
    }
    friend int wrap(vec & v, const vec & w) {
      int iw = 0;
      for (int i=0; i<N; i++) {
	if(v[i] < -w[i] / 2) {
	  v[i] += w[i];
	  iw |= 1 << i;
	}
	if(v[i] >  w[i] / 2) {
	  v[i] -= w[i];
	  iw |= 1 << i;
	}
      }
      return iw;
    }
    friend void unwrap(vec & v, const vec & w, const int & iw) {
      for (int i=0; i<N; i++) {
	if((iw >> i) & 1) v[i] += (v[i] > 0 ? -w[i] : w[i]);
      }
    }
  };
#else
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for CUDA")
#endif
#include "unroll.h"
  template<int N, typename T>
  class vec {
  private:
    T data[N];
  public:
    __host__ __device__ __forceinline__
    vec(){}
    __host__ __device__ __forceinline__
    vec(const T &v) {
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
    }
    __host__ __device__ __forceinline__
    vec(const vec &v) {
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
    }
    __host__ __device__ __forceinline__
    vec(const float4 &v) {
      data[0] = v.x;
      data[1] = v.y;
      data[2] = v.z;
      data[3] = v.w;
    }
    __host__ __device__ __forceinline__
    vec(const float x, const float y, const float z, const float w) {
      data[0] = x;
      data[1] = y;
      data[2] = z;
      data[3] = w;
    }
    __host__ __device__ __forceinline__
    vec(const float x, const float y, const float z) {
      data[0] = x;
      data[1] = y;
      data[2] = z;
    }
    __host__ __device__ __forceinline__
    ~vec(){}
    __host__ __device__ __forceinline__
    const vec &operator=(const T v) {
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator+=(const T v) {
      Unroll<Ops::Add<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator-=(const T v) {
      Unroll<Ops::Sub<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator*=(const T v) {
      Unroll<Ops::Mul<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator/=(const T v) {
      Unroll<Ops::Div<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator>=(const T v) {
      Unroll<Ops::Gt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator<=(const T v) {
      Unroll<Ops::Lt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator&=(const T v) {
      Unroll<Ops::And<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator|=(const T v) {
      Unroll<Ops::Or<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator=(const vec & v) {
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator+=(const vec & v) {
      Unroll<Ops::Add<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator-=(const vec & v) {
      Unroll<Ops::Sub<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator*=(const vec & v) {
      Unroll<Ops::Mul<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator/=(const vec & v) {
      Unroll<Ops::Div<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator>=(const vec & v) {
      Unroll<Ops::Gt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator<=(const vec & v) {
      Unroll<Ops::Lt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator&=(const vec & v) {
      Unroll<Ops::And<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator|=(const vec & v) {
      Unroll<Ops::Or<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    vec operator+(const T v) const {
      return vec(*this) += v;
    }
    __host__ __device__ __forceinline__
    vec operator-(const T v) const {
      return vec(*this) -= v;
    }
    __host__ __device__ __forceinline__
    vec operator*(const T v) const {
      return vec(*this) *= v;
    }
    __host__ __device__ __forceinline__
    vec operator/(const T v) const {
      return vec(*this) /= v;
    }
    __host__ __device__ __forceinline__
    vec operator>(const T v) const {
      return vec(*this) >= v;
    }
    __host__ __device__ __forceinline__
    vec operator<(const T v) const {
      return vec(*this) <= v;
    }
    __host__ __device__ __forceinline__
    vec operator&(const T v) const {
      return vec(*this) &= v;
    }
    __host__ __device__ __forceinline__
    vec operator|(const T v) const {
      return vec(*this) |= v;
    }
    __host__ __device__ __forceinline__
    vec operator+(const vec & v) const {
      return vec(*this) += v;
    }
    __host__ __device__ __forceinline__
    vec operator-(const vec & v) const {
      return vec(*this) -= v;
    }
    __host__ __device__ __forceinline__
    vec operator*(const vec & v) const {
      return vec(*this) *= v;
    }
    __host__ __device__ __forceinline__
    vec operator/(const vec & v) const {
      return vec(*this) /= v;
    }
    __host__ __device__ __forceinline__
    vec operator>(const vec & v) const {
      return vec(*this) >= v;
    }
    __host__ __device__ __forceinline__
    vec operator<(const vec & v) const {
      return vec(*this) <= v;
    }
    __host__ __device__ __forceinline__
    vec operator&(const vec & v) const {
      return vec(*this) &= v;
    }
    __host__ __device__ __forceinline__
    vec operator|(const vec & v) const {
      return vec(*this) |= v;
    }
    __host__ __device__ __forceinline__
    vec operator-() const {
      vec temp;
      Unroll<Ops::Negate<T>,T,N>::loop(temp,data);
      return temp;
    }
    __host__ __device__ __forceinline__
    T &operator[](int i) {
      return data[i];
    }
    __host__ __device__ __forceinline__
    const T &operator[](int i) const {
      return data[i];
    }
    __host__ __device__ __forceinline__
    operator       T* ()       {return data;}
    __host__ __device__ __forceinline__
    friend T min(const vec & v) {
      T temp;
      for (int i=0; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T max(const vec & v) {
      T temp;
      for (int i=0; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }  __host__ __device__ __forceinline__
    operator const T* () const {return data;}
    __host__ __device__ __forceinline__
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<N; i++) s << v[i] << ' ';
      return s;
    }
    __host__ __device__ __forceinline__
    friend T sum(const vec & v) {
      return Unroll<Ops::Add<T>,T,N>::reduce(v);
    }
    __host__ __device__ __forceinline__
    friend T norm(const vec & v) {
      return sum(v * v);
    }
    __host__ __device__ __forceinline__
    friend vec min(const vec & v, const vec & w) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] < w[i] ? v[i] : w[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec max(const vec & v, const vec & w) {
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] > w[i] ? v[i] : w[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T min(const vec & v) {
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T max(const vec & v) {
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }
    __device__ __forceinline__
    friend vec abs(const vec & v) {
      vec temp;
      Unroll<Ops::Abs<T>,T,N>::loop(temp,v);
      return temp;
    }
    __device__ __forceinline__
    friend vec rsqrt(const vec & v) {
      vec temp;
      Unroll<Ops::Rsqrt<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec sin(const vec & v) {
      vec temp;
      Unroll<Ops::Sin<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec cos(const vec & v) {
      vec temp;
      Unroll<Ops::Cos<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend void sincos(vec & s, vec & c, const vec & v) {
      Unroll<Ops::SinCos<T>,T,N>::loop(s,c,v);
    }
    __host__ __device__ __forceinline__
    friend vec exp(const vec & v) {
      vec temp;
      Unroll<Ops::Exp<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend int wrap(vec & v, const T & w) {
      int iw = 0;
      for (int i=0; i<N; i++) {
	if(v[i] < -w / 2) {
	  v[i] += w;
	  iw |= 1 << i;
	}
	if(v[i] >  w / 2) {
	  v[i] -= w;
	  iw |= 1 << i;
	}
      }
      return iw;
    }
    __host__ __device__ __forceinline__
    friend void unwrap(vec & v, const T & w, const int & iw) {
      for (int i=0; i<N; i++) {
	if((iw >> i) & 1) v[i] += (v[i] > 0 ? -w : w);
      }
    }
  };
#endif

#if defined __MIC__ || defined __AVX512F__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for AVX512/MIC")
#endif
#include <immintrin.h>
  template<>
  class vec<16,float> {
  private:
    union {
      __m512 data;
      float array[16];
    };
  public:
    vec(){}
    vec(const float v) {
      data = _mm512_set1_ps(v);
    }
    vec(const __m512 v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d,
	const float e, const float f, const float g, const float h,
	const float i, const float j, const float k, const float l,
	const float m, const float n, const float o, const float p) {
      data = _mm512_setr_ps(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);
    }
    vec(const float* a, const int size) {
      int offset = size / (int)sizeof(float);
      data = _mm512_setr_ps(*a, *(a + 1 * offset), *(a + 2 * offset), *(a + 3 * offset), *(a + 4 * offset), *(a + 5 * offset), *(a + 6 * offset), *(a + 7 * offset), *(a + 8 * offset), *(a + 9 * offset), *(a + 10 * offset), *(a + 11 * offset), *(a + 12 * offset), *(a + 13 * offset), *(a + 14 * offset), *(a + 15 * offset));
    }
    ~vec(){}
    const vec &operator=(const float v) {
      data = _mm512_set1_ps(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm512_add_ps(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm512_sub_ps(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm512_mul_ps(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm512_div_ps(data,v.data);
      return *this;
    }
    const vec &operator&=(const __mmask16 & v) {
      data = _mm512_mask_mov_ps(_mm512_setzero_ps(),v,data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm512_add_ps(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm512_sub_ps(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm512_mul_ps(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm512_div_ps(data,v.data));
    }
    __mmask16 operator>(const vec & v) const {
      return _mm512_cmp_ps_mask(data,v.data,_MM_CMPINT_GT);
    }
    __mmask16 operator<(const vec & v) const {
      return _mm512_cmp_ps_mask(data,v.data,_MM_CMPINT_LT);
    }
    vec operator-() const {
      return vec(_mm512_sub_ps(_mm512_setzero_ps(),data));
    }
    float &operator[](int i) {
      return array[i];
    }
    const float &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<16; i++) s << v[i] << ' ';
      return s;
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm512_min_ps(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm512_max_ps(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
      vec three = 3.0;
      vec twelve = 12.0;
#ifdef __MIC__
      vec temp = vec(_mm512_rsqrt23_ps(v.data));
#else
      vec temp = vec(_mm512_rsqrt14_ps(v.data));
#endif
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm512_div_ps(one.data,_mm512_sqrt_ps(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec & v) {
      return vec(_mm512_sin_ps(v.data));
    }
    friend vec cos(const vec & v) {
      return vec(_mm512_cos_ps(v.data));
    }
    friend vec exp(const vec & v) {
      return vec(_mm512_exp_ps(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm512_setr_ps(std::sin(v[0]), std::sin(v[1]), std::sin(v[2]), std::sin(v[3]),
                                std::sin(v[4]), std::sin(v[5]), std::sin(v[6]), std::sin(v[7]),
                                std::sin(v[8]), std::sin(v[9]), std::sin(v[10]), std::sin(v[11]),
                                std::sin(v[12]), std::sin(v[13]), std::sin(v[14]), std::sin(v[15]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm512_setr_ps(std::cos(v[0]), std::cos(v[1]), std::cos(v[2]), std::cos(v[3]),
                                std::cos(v[4]), std::cos(v[5]), std::cos(v[6]), std::cos(v[7]),
                                std::cos(v[8]), std::cos(v[9]), std::cos(v[10]), std::cos(v[11]),
                                std::cos(v[12]), std::cos(v[13]), std::cos(v[14]), std::cos(v[15]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm512_setr_ps(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]), std::exp(v[3]),
                                std::exp(v[4]), std::exp(v[5]), std::exp(v[6]), std::exp(v[7]),
                                std::exp(v[8]), std::exp(v[9]), std::exp(v[10]), std::exp(v[11]),
                                std::exp(v[12]), std::exp(v[13]), std::exp(v[14]), std::exp(v[15]));
      return temp;
    }
#endif
  };

  template<>
  class vec<8,double> {
  private:
    union {
      __m512d data;
      double array[8];
    };
  public:
    vec(){}
    vec(const double v) {
      data = _mm512_set1_pd(v);
    }
    vec(const __m512d v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const double a, const double b, const double c, const double d,
	const double e, const double f, const double g, const double h) {
      data = _mm512_setr_pd(a,b,c,d,e,f,g,h);
    }
    vec(const double* a) {
      data = _mm512_setr_pd(*a, *(a + 1), *(a + 2), *(a + 3), *(a + 4), *(a + 5), *(a + 6), *(a + 7));
    }
    vec(const double* a, const int size) {
      int offset = size / (int)sizeof(double);
      data = _mm512_setr_pd(*a, *(a + 1 * offset), *(a + 2 * offset), *(a + 3 * offset), *(a + 4 * offset), *(a + 5 * offset), *(a + 6 * offset), *(a + 7 * offset));
    }
    ~vec(){}
    const vec &operator=(const double v) {
      data = _mm512_set1_pd(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm512_add_pd(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm512_sub_pd(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm512_mul_pd(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm512_div_pd(data,v.data);
      return *this;
    }
    const vec &operator&=(const __mmask8 & v) {
      data = _mm512_mask_mov_pd(_mm512_setzero_pd(),v,data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm512_add_pd(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm512_sub_pd(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm512_mul_pd(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm512_div_pd(data,v.data));
    }
    __mmask8 operator>(const vec & v) const {
      return _mm512_cmp_pd_mask(data,v.data,_MM_CMPINT_GT);
    }
    __mmask8 operator<(const vec & v) const {
      return _mm512_cmp_pd_mask(data,v.data,_MM_CMPINT_LT);
    }
    vec operator-() const {
      return vec(_mm512_sub_pd(_mm512_setzero_pd(),data));
    }
    double &operator[](int i) {
      return array[i];
    }
    const double &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<8; i++) s << v[i] << ' ';
      return s;
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm512_min_pd(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm512_max_pd(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
#ifdef __MIC__
      vec temp = vec(_mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps(v.data))));
#else
      vec temp = vec(_mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps(v.data))));
#endif
      vec three = 3.0;
      vec twelve = 12.0;
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm512_div_pd(one.data,_mm512_sqrt_pd(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec & v) {
      return vec(_mm512_sin_pd(v.data));
    }
    friend vec cos(const vec & v) {
      return vec(_mm512_cos_pd(v.data));
    }
    friend vec exp(const vec & v) {
      return vec(_mm512_exp_pd(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm512_setr_pd(std::sin(v[0]), std::sin(v[1]), std::sin(v[2]), std::sin(v[3]),
                                std::sin(v[4]), std::sin(v[5]), std::sin(v[6]), std::sin(v[7]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm512_setr_pd(std::cos(v[0]), std::cos(v[1]), std::cos(v[2]), std::cos(v[3]),
                                std::cos(v[4]), std::cos(v[5]), std::cos(v[6]), std::cos(v[7]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm512_setr_pd(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]), std::exp(v[3]),
                                std::exp(v[4]), std::exp(v[5]), std::exp(v[6]), std::exp(v[7]));
      return temp;
    }
#endif
  };
#endif

#ifdef __AVX__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for AVX")
#endif
#include <immintrin.h>

  template<>
  class vec<8,float> {
  private:
    union {
      __m256 data;
      float array[8];
    };
  public:
    vec(){}
    vec(const float v) {
      data = _mm256_set1_ps(v);
    }
    vec(const __m256 v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d,
        const float e, const float f, const float g, const float h) {
      data = _mm256_setr_ps(a,b,c,d,e,f,g,h);
    }
    vec(const float* a, const int size) {
      int offset = size / (int)sizeof(float);
      data = _mm256_setr_ps(*a, *(a + 1 * offset), *(a + 2 * offset), *(a + 3 * offset), *(a + 4 * offset), *(a + 5 * offset), *(a + 6 * offset), *(a + 7 * offset));
    }
    ~vec(){}
    const vec &operator=(const float v) {
      data = _mm256_set1_ps(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm256_add_ps(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm256_sub_ps(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm256_mul_ps(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm256_div_ps(data,v.data);
      return *this;
    }
    const vec &operator&=(const vec & v) {
      data = _mm256_and_ps(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm256_add_ps(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm256_sub_ps(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm256_mul_ps(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm256_div_ps(data,v.data));
    }
    vec operator>(const vec & v) const {
      return vec(_mm256_cmp_ps(data,v.data,_CMP_GT_OQ));
    }
    vec operator<(const vec & v) const {
      return vec(_mm256_cmp_ps(data,v.data,_CMP_LT_OQ));
    }
    vec operator-() const {
      return vec(_mm256_sub_ps(_mm256_setzero_ps(),data));
    }
    float &operator[](int i) {
      return array[i];
    }
    const float &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<8; i++) s << v[i] << ' ';
      return s;
    }
    friend float sum(const vec & v) {
      union {
        __m256 temp;
        float out[8];
      };
      temp = _mm256_permute2f128_ps(v.data,v.data,1);
      temp = _mm256_add_ps(temp,v.data);
      temp = _mm256_hadd_ps(temp,temp);
      temp = _mm256_hadd_ps(temp,temp);
      return out[0];
    }
    friend float norm(const vec & v) {
      union {
        __m256 temp;
        float out[8];
      };
      temp = _mm256_mul_ps(v.data,v.data);
      __m256 perm = _mm256_permute2f128_ps(temp,temp,1);
      temp = _mm256_add_ps(temp,perm);
      temp = _mm256_hadd_ps(temp,temp);
      temp = _mm256_hadd_ps(temp,temp);
      return out[0];
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm256_min_ps(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm256_max_ps(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
      vec three = 3.0f;
      vec twelve = 12.0f;
      vec temp = vec(_mm256_rsqrt_ps(v.data));
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm256_div_ps(one.data,_mm256_sqrt_ps(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec & v) {
      return vec(_mm256_sin_ps(v.data));
    }
    friend vec cos(const vec & v) {
      return vec(_mm256_cos_ps(v.data));
    }
    friend vec exp(const vec & v) {
      return vec(_mm256_exp_ps(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm256_setr_ps(std::sin(v[0]), std::sin(v[1]), std::sin(v[2]), std::sin(v[3]),
                                std::sin(v[4]), std::sin(v[5]), std::sin(v[6]), std::sin(v[7]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm256_setr_ps(std::cos(v[0]), std::cos(v[1]), std::cos(v[2]), std::cos(v[3]),
                                std::cos(v[4]), std::cos(v[5]), std::cos(v[6]), std::cos(v[7]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm256_setr_ps(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]), std::exp(v[3]),
                                std::exp(v[4]), std::exp(v[5]), std::exp(v[6]), std::exp(v[7]));
      return temp;
    }
#endif
  };

  template<>
  class vec<4,double> {
  private:
    union {
      __m256d data;
      double array[4];
    };
  public:
    vec(){}
    vec(const double v) {
      data = _mm256_set1_pd(v);
    }
    vec(const __m256d v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const double a, const double b, const double c, const double d) {
      data = _mm256_setr_pd(a,b,c,d);
    }
    vec(const double* a) {
      data = _mm256_setr_pd(*a, *(a + 1), *(a + 2), *(a + 3));
    }
    vec(const double* a, const int size) {
      int offset = size / (int)sizeof(double);
      data = _mm256_setr_pd(*a, *(a + 1 * offset), *(a + 2 * offset), *(a + 3 * offset));
    }
    ~vec(){}
    const vec &operator=(const double v) {
      data = _mm256_set1_pd(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm256_add_pd(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm256_sub_pd(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm256_mul_pd(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm256_div_pd(data,v.data);
      return *this;
    }
    const vec &operator&=(const vec & v) {
      data = _mm256_and_pd(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm256_add_pd(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm256_sub_pd(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm256_mul_pd(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm256_div_pd(data,v.data));
    }
    vec operator>(const vec & v) const {
      return vec(_mm256_cmp_pd(data,v.data,_CMP_GT_OQ));
    }
    vec operator<(const vec & v) const {
      return vec(_mm256_cmp_pd(data,v.data,_CMP_LT_OQ));
    }
    vec operator-() const {
      return vec(_mm256_sub_pd(_mm256_setzero_pd(),data));
    }
    double &operator[](int i) {
      return array[i];
    }
    const double &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<4; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {
      union {
        __m256d temp;
        double out[4];
      };
      temp = _mm256_permute2f128_pd(v.data,v.data,1);
      temp = _mm256_add_pd(temp,v.data);
      temp = _mm256_hadd_pd(temp,temp);
      return out[0];
    }
    friend double norm(const vec & v) {
      union {
        __m256d temp;
        double out[4];
      };
      temp = _mm256_mul_pd(v.data,v.data);
      __m256d perm = _mm256_permute2f128_pd(temp,temp,1);
      temp = _mm256_add_pd(temp,perm);
      temp = _mm256_hadd_pd(temp,temp);
      return out[0];
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm256_min_pd(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm256_max_pd(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
      vec temp = vec(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(v.data))));
      vec three = 3.0;
      vec twelve = 12.0;
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm256_div_pd(one.data,_mm256_sqrt_pd(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec & v) {
      return vec(_mm256_sin_pd(v.data));
    }
    friend vec cos(const vec & v) {
      return vec(_mm256_cos_pd(v.data));
    }
    friend vec exp(const vec & v) {
      return vec(_mm256_exp_pd(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm256_setr_pd(std::sin(v[0]), std::sin(v[1]), std::sin(v[2]), std::sin(v[3]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm256_setr_pd(std::cos(v[0]), std::cos(v[1]), std::cos(v[2]), std::cos(v[3]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm256_setr_pd(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]), std::exp(v[3]));
      return temp;
    }
#endif
  };
#endif

#ifdef __SSE__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for SSE")
#endif
#include <pmmintrin.h>

  template<>
  class vec<4,float> {
  private:
    union {
      __m128 data;
      float array[4];
    };
  public:
    vec(){}
    vec(const float v) {
      data = _mm_set1_ps(v);
    }
    vec(const __m128 v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d) {
      data = _mm_setr_ps(a,b,c,d);
    }
    vec(const float* a, const int size) {
      int offset = size / (int)sizeof(float);
      data = _mm_setr_ps(*a, *(a + 1 * offset), *(a + 2 * offset), *(a + 3 * offset));
    }
    ~vec(){}
    const vec &operator=(const float v) {
      data = _mm_set1_ps(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm_add_ps(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm_sub_ps(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm_mul_ps(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm_div_ps(data,v.data);
      return *this;
    }
    const vec &operator&=(const vec & v) {
      data = _mm_and_ps(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm_add_ps(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm_sub_ps(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm_mul_ps(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm_div_ps(data,v.data));
    }
    vec operator>(const vec & v) const {
      return vec(_mm_cmpgt_ps(data,v.data));
    }
    vec operator<(const vec & v) const {
      return vec(_mm_cmplt_ps(data,v.data));
    }
    vec operator-() const {
      return vec(_mm_sub_ps(_mm_setzero_ps(),data));
    }
    float &operator[](int i) {
      return array[i];
    }
    const float &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<4; i++) s << v[i] << ' ';
      return s;
    }
    friend float sum(const vec & v) {
      union {
        __m128 temp;
        float out[4];
      };
      temp = _mm_hadd_ps(v.data,v.data);
      temp = _mm_hadd_ps(temp,temp);
      return out[0];
    }
    friend float norm(const vec & v) {
      union {
        __m128 temp;
        float out[4];
      };
      temp = _mm_mul_ps(v.data,v.data);
      temp = _mm_hadd_ps(temp,temp);
      temp = _mm_hadd_ps(temp,temp);
      return out[0];
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm_min_ps(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm_max_ps(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
      vec temp = vec(_mm_rsqrt_ps(v.data));
      vec three = 3.0f;
      vec twelve = 12.0f;
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm_div_ps(one.data,_mm_sqrt_ps(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec &v) {
      return vec(_mm_sin_ps(v.data));
    }
    friend vec cos(const vec &v) {
      return vec(_mm_cos_ps(v.data));
    }
    friend vec exp(const vec &v) {
      return vec(_mm_exp_ps(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm_setr_ps(std::sin(v[0]), std::sin(v[1]), std::sin(v[2]), std::sin(v[3]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm_setr_ps(std::cos(v[0]), std::cos(v[1]), std::cos(v[2]), std::cos(v[3]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm_setr_ps(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]), std::exp(v[3]));
      return temp;
    }
#endif
  };

  template<>
  class vec<2,double> {
  private:
    union {
      __m128d data;
      double array[2];
    };
  public:
    vec(){}
    vec(const double v) {
      data = _mm_set1_pd(v);
    }
    vec(const __m128d v) {
      data = v;
    }
    vec(const vec & v) {
      data = v.data;
    }
    vec(const double a, const double b) {
      data = _mm_setr_pd(a,b);
    }
    vec(const double* a, const int size) {
      int offset = size / (int)sizeof(double);
      data = _mm_setr_pd(*a, *(a + 1 * offset));
    }
    ~vec(){}
    const vec &operator=(const double v) {
      data = _mm_set1_pd(v);
      return *this;
    }
    const vec &operator=(const vec & v) {
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {
      data = _mm_add_pd(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {
      data = _mm_sub_pd(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {
      data = _mm_mul_pd(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {
      data = _mm_div_pd(data,v.data);
      return *this;
    }
    const vec &operator&=(const vec & v) {
      data = _mm_and_pd(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {
      return vec(_mm_add_pd(data,v.data));
    }
    vec operator-(const vec & v) const {
      return vec(_mm_sub_pd(data,v.data));
    }
    vec operator*(const vec & v) const {
      return vec(_mm_mul_pd(data,v.data));
    }
    vec operator/(const vec & v) const {
      return vec(_mm_div_pd(data,v.data));
    }
    vec operator>(const vec & v) const {
      return vec(_mm_cmpgt_pd(data,v.data));
    }
    vec operator<(const vec & v) const {
      return vec(_mm_cmplt_pd(data,v.data));
    }
    vec operator-() const {
      return vec(_mm_sub_pd(_mm_setzero_pd(),data));
    }
    double &operator[](int i) {
      return array[i];
    }
    const double &operator[](int i) const {
      return array[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {
      for (int i=0; i<2; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {
      union {
        __m128d temp;
        double out[2];
      };
      temp = _mm_hadd_pd(v.data,v.data);
      return out[0];
    }
    friend double norm(const vec & v) {
      union {
        __m128d temp;
        double out[2];
      };
      temp = _mm_mul_pd(v.data,v.data);
      temp = _mm_hadd_pd(temp,temp);
      return out[0];
    }
    friend vec min(const vec & v, const vec & w) {
      return vec(_mm_min_pd(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {
      return vec(_mm_max_pd(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {
#if EXAFMM_RSQRT_APPROX
      vec temp = vec(_mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(v.data))));
      vec three = 3.0;
      vec twelve = 12.0;
      temp *= (three - temp * temp * v);
      temp *= (twelve - temp * temp * v);
      return temp;
#else
      vec one = 1;
      return vec(_mm_div_pd(one.data,_mm_sqrt_pd(v.data)));
#endif
    }
#ifdef __INTEL_COMPILER
    friend vec sin(const vec &v) {
      return vec(_mm_sin_pd(v.data));
    }
    friend vec cos(const vec &v) {
      return vec(_mm_cos_pd(v.data));
    }
    friend vec exp(const vec &v) {
      return vec(_mm_exp_pd(v.data));
    }
#else
    friend vec sin(const vec & v) {
      vec temp = _mm_setr_pd(std::sin(v[0]), std::sin(v[1]));
      return temp;
    }
    friend vec cos(const vec & v) {
      vec temp = _mm_setr_pd(std::cos(v[0]), std::cos(v[1]));
      return temp;
    }
    friend vec exp(const vec & v) {
      vec temp = _mm_setr_pd(std::exp(v[0]), std::exp(v[1]));
      return temp;
    }
#endif
  };
#endif

}
#endif
