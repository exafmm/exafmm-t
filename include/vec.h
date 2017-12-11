#ifndef vec_h
#define vec_h
#include <ostream>
#define EXAFMM_VEC_NEWTON 1
#define EXAFMM_VEC_VERBOSE 0
#if EXAFMM_VEC_VERBOSE
#pragma message("Using Agner's vectorclass for operator overloading SIMD intrinsics.")
#endif
#include "../vectorclass/vectorclass.h"
#include "../vectorclass/vectormath_common.h"
#include "../vectorclass/vectormath_exp.h"
#include "../vectorclass/vectormath_trig.h"
//! Custom vector type for small vectors with template specialization for MIC, AVX, SSE intrinsics

namespace exafmm {
#ifndef __CUDACC__
  template<int N, typename T>
  class vec {
  private:
    T data[N];
  public:
    vec(){}                                                     // Default constructor
    vec(const T &v) {                                           // Copy constructor (scalar)
      for (int i=0; i<N; i++) data[i] = v;
    }
    vec(const vec &v) {                                         // Copy constructor (vector)
      for (int i=0; i<N; i++) data[i] = v[i];
    }
    ~vec(){}                                                    // Destructor
    const vec &operator=(const T v) {                           // Scalar assignment
      for (int i=0; i<N; i++) data[i] = v;
      return *this;
    }
    const vec &operator+=(const T v) {                          // Scalar compound assignment (add)
      for (int i=0; i<N; i++) data[i] += v;
      return *this;
    }
    const vec &operator-=(const T v) {                          // Scalar compound assignment (subtract)
      for (int i=0; i<N; i++) data[i] -= v;
      return *this;
    }
    const vec &operator*=(const T v) {                          // Scalar compound assignment (multiply)
      for (int i=0; i<N; i++) data[i] *= v;
      return *this;
    }
    const vec &operator/=(const T v) {                          // Scalar compound assignment (divide)
      for (int i=0; i<N; i++) data[i] /= v;
      return *this;
    }
    const vec &operator>=(const T v) {                          // Scalar compound assignment (greater than)
      for (int i=0; i<N; i++) data[i] >= v;
      return *this;
    }
    const vec &operator<=(const T v) {                          // Scalar compound assignment (less than)
      for (int i=0; i<N; i++) data[i] <= v;
      return *this;
    }
    const vec &operator&=(const T v) {                          // Scalar compound assignment (bitwise and)
      for (int i=0; i<N; i++) data[i] &= v;
      return *this;
    }
    const vec &operator|=(const T v) {                          // Scalar compound assignment (bitwise or)
      for (int i=0; i<N; i++) data[i] |= v;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      for (int i=0; i<N; i++) data[i] = v[i];
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      for (int i=0; i<N; i++) data[i] += v[i];
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      for (int i=0; i<N; i++) data[i] -= v[i];
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      for (int i=0; i<N; i++) data[i] *= v[i];
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      for (int i=0; i<N; i++) data[i] /= v[i];
      return *this;
    }
    const vec &operator>=(const vec & v) {                      // Vector compound assignment (greater than)
      for (int i=0; i<N; i++) data[i] >= v[i];
      return *this;
    }
    const vec &operator<=(const vec & v) {                      // Vector compound assignment (less than)
      for (int i=0; i<N; i++) data[i] <= v[i];
      return *this;
    }
    const vec &operator&=(const vec & v) {                      // Vector compound assignment (bitwise and)
      for (int i=0; i<N; i++) data[i] &= v[i];
      return *this;
    }
    const vec &operator|=(const vec & v) {                      // Vector compound assignment (bitwise or)
      for (int i=0; i<N; i++) data[i] |= v[i];
      return *this;
    }
    vec operator+(const T v) const {                            // Scalar arithmetic (add)
      return vec(*this) += v;
    }
    vec operator-(const T v) const {                            // Scalar arithmetic (subtract)
      return vec(*this) -= v;
    }
    vec operator*(const T v) const {                            // Scalar arithmetic (multiply)
      return vec(*this) *= v;
    }
    vec operator/(const T v) const {                            // Scalar arithmetic (divide)
      return vec(*this) /= v;
    }
    vec operator>(const T v) const {                            // Scalar arithmetic (greater than)
      return vec(*this) >= v;
    }
    vec operator<(const T v) const {                            // Scalar arithmetic (less than)
      return vec(*this) <= v;
    }
    vec operator&(const T v) const {                            // Scalar arithmetic (bitwise and)
      return vec(*this) &= v;
    }
    vec operator|(const T v) const {                            // Scalar arithmetic (bitwise or)
      return vec(*this) |= v;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(*this) += v;
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(*this) -= v;
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(*this) *= v;
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(*this) /= v;
    }
    vec operator>(const vec & v) const {                        // Vector arithmetic (greater than)
      return vec(*this) >= v;
    }
    vec operator<(const vec & v) const {                        // Vector arithmetic (less than)
      return vec(*this) <= v;
    }
    vec operator&(const vec & v) const {                        // Vector arithmetic (bitwise and)
      return vec(*this) &= v;
    }
    vec operator|(const vec & v) const {                        // Vector arithmetic (bitwise or)
      return vec(*this) |= v;
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      vec temp;
      for (int i=0; i<N; i++) temp[i] = -data[i];
      return temp;
    }
    T &operator[](int i) {                                      // Indexing (lvalue)
      return data[i];
    }
    const T &operator[](int i) const {                          // Indexing (rvalue)
      return data[i];
    }
    operator       T* ()       {return data;}                   // Type-casting (lvalue)
    operator const T* () const {return data;}                   // Type-casting (rvalue)
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<N; i++) s << v[i] << ' ';
      return s;
    }
    friend T sum(const vec & v) {                               // Sum vector
      T temp = 0;
      for (int i=0; i<N; i++) temp += v[i];
      return temp;
    }
    friend T norm(const vec & v) {                              // L2 norm squared
      T temp = 0;
      for (int i=0; i<N; i++) temp += v[i] * v[i];
      return temp;
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] < w[i] ? v[i] : w[i];
      return temp;
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] > w[i] ? v[i] : w[i];
      return temp;
    }
    friend T min(const vec & v) {                               // Reduce minimum
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    friend T max(const vec & v) {                               // Reduce maximum
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }
    friend vec sin(const vec & v) {                             // Sine function
      vec temp;
      for (int i=0; i<N; i++) temp[i] = sin(v[i]);
      return temp;
    }
    friend vec cos(const vec & v) {                             // Cosine function
      vec temp;
      for (int i=0; i<N; i++) temp[i] = cos(v[i]);
      return temp;
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      for (int i=0; i<N; i++) {
	s[i] = sin(v[i]);
	c[i] = cos(v[i]);
      }
    }
    friend vec exp(const vec & v) {                             // Exponential function
      vec temp;
      for (int i=0; i<N; i++) temp[i] = exp(v[i]);
      return temp;
    }
    friend int wrap(vec & v, const vec & w) {                     // Wrap around periodic boundary
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
    friend void unwrap(vec & v, const vec & w, const int & iw) {  // Undo wrap around periodic boundary
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
    vec(){}                                                     // Default constructor
    __host__ __device__ __forceinline__
    vec(const T &v) {                                           // Copy constructor (scalar)
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
    }
    __host__ __device__ __forceinline__
    vec(const vec &v) {                                         // Copy constructor (vector)
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
    }
    __host__ __device__ __forceinline__
    vec(const float4 &v) {                                      // Copy constructor (float4)
      data[0] = v.x;
      data[1] = v.y;
      data[2] = v.z;
      data[3] = v.w;
    }
    __host__ __device__ __forceinline__
    vec(const float x, const float y, const float z, const float w) {// Copy constructor (4 floats)
      data[0] = x;
      data[1] = y;
      data[2] = z;
      data[3] = w;
    }
    __host__ __device__ __forceinline__
    vec(const float x, const float y, const float z) {          // Copy constructor (3 floats)
      data[0] = x;
      data[1] = y;
      data[2] = z;
    }
    __host__ __device__ __forceinline__
    ~vec(){}                                                    // Destructor
    __host__ __device__ __forceinline__
    const vec &operator=(const T v) {                           // Scalar assignment
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator+=(const T v) {                          // Scalar compound assignment (add)
      Unroll<Ops::Add<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator-=(const T v) {                          // Scalar compound assignment (subtract)
      Unroll<Ops::Sub<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator*=(const T v) {                          // Scalar compound assignment (multiply)
      Unroll<Ops::Mul<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator/=(const T v) {                          // Scalar compound assignment (divide)
      Unroll<Ops::Div<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator>=(const T v) {                          // Scalar compound assignment (greater than)
      Unroll<Ops::Gt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator<=(const T v) {                          // Scalar compound assignment (less than)
      Unroll<Ops::Lt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator&=(const T v) {                          // Scalar compound assignment (bitwise and)
      Unroll<Ops::And<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator|=(const T v) {                          // Scalar compound assignment (bitwise or)
      Unroll<Ops::Or<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator=(const vec & v) {                       // Vector assignment
      Unroll<Ops::Assign<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      Unroll<Ops::Add<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      Unroll<Ops::Sub<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      Unroll<Ops::Mul<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      Unroll<Ops::Div<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator>=(const vec & v) {                      // Vector compound assignment (greater than)
      Unroll<Ops::Gt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator<=(const vec & v) {                      // Vector compound assignment (less than)
      Unroll<Ops::Lt<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator&=(const vec & v) {                      // Vector compound assignment (bitwise and)
      Unroll<Ops::And<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    const vec &operator|=(const vec & v) {                      // Vector compound assignment (bitwise or)
      Unroll<Ops::Or<T>,T,N>::loop(data,v);
      return *this;
    }
    __host__ __device__ __forceinline__
    vec operator+(const T v) const {                            // Scalar arithmetic (add)
      return vec(*this) += v;
    }
    __host__ __device__ __forceinline__
    vec operator-(const T v) const {                            // Scalar arithmetic (subtract)
      return vec(*this) -= v;
    }
    __host__ __device__ __forceinline__
    vec operator*(const T v) const {                            // Scalar arithmetic (multiply)
      return vec(*this) *= v;
    }
    __host__ __device__ __forceinline__
    vec operator/(const T v) const {                            // Scalar arithmetic (divide)
      return vec(*this) /= v;
    }
    __host__ __device__ __forceinline__
    vec operator>(const T v) const {                            // Scalar arithmetic (greater than)
      return vec(*this) >= v;
    }
    __host__ __device__ __forceinline__
    vec operator<(const T v) const {                            // Scalar arithmetic (less than)
      return vec(*this) <= v;
    }
    __host__ __device__ __forceinline__
    vec operator&(const T v) const {                            // Scalar arithmetic (bitwise and)
      return vec(*this) &= v;
    }
    __host__ __device__ __forceinline__
    vec operator|(const T v) const {                            // Scalar arithmetic (bitwise or)
      return vec(*this) |= v;
    }
    __host__ __device__ __forceinline__
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(*this) += v;
    }
    __host__ __device__ __forceinline__
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(*this) -= v;
    }
    __host__ __device__ __forceinline__
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(*this) *= v;
    }
    __host__ __device__ __forceinline__
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(*this) /= v;
    }
    __host__ __device__ __forceinline__
    vec operator>(const vec & v) const {                        // Vector arithmetic (greater than)
      return vec(*this) >= v;
    }
    __host__ __device__ __forceinline__
    vec operator<(const vec & v) const {                        // Vector arithmetic (less than)
      return vec(*this) <= v;
    }
    __host__ __device__ __forceinline__
    vec operator&(const vec & v) const {                        // Vector arithmetic (bitwise and)
      return vec(*this) &= v;
    }
    __host__ __device__ __forceinline__
    vec operator|(const vec & v) const {                        // Vector arithmetic (bitwise or)
      return vec(*this) |= v;
    }
    __host__ __device__ __forceinline__
    vec operator-() const {                                     // Vector arithmetic (negation)
      vec temp;
      Unroll<Ops::Negate<T>,T,N>::loop(temp,data);
      return temp;
    }
    __host__ __device__ __forceinline__
    T &operator[](int i) {                                      // Indexing (lvalue)
      return data[i];
    }
    __host__ __device__ __forceinline__
    const T &operator[](int i) const {                          // Indexing (rvalue)
      return data[i];
    }
    __host__ __device__ __forceinline__
    operator       T* ()       {return data;}                   // Type-casting (lvalue)
    __host__ __device__ __forceinline__
    friend T min(const vec & v) {                               // Reduce minimum
      T temp;
      for (int i=0; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T max(const vec & v) {                               // Reduce maximum
      T temp;
      for (int i=0; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }  __host__ __device__ __forceinline__
    operator const T* () const {return data;}                   // Type-casting (rvalue)
    __host__ __device__ __forceinline__
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<N; i++) s << v[i] << ' ';
      return s;
    }
    __host__ __device__ __forceinline__
    friend T sum(const vec & v) {                               // Sum vector
      return Unroll<Ops::Add<T>,T,N>::reduce(v);
    }
    __host__ __device__ __forceinline__
    friend T norm(const vec & v) {                              // L2 norm squared
      return sum(v * v);
    }
    __host__ __device__ __forceinline__
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] < w[i] ? v[i] : w[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      vec temp;
      for (int i=0; i<N; i++) temp[i] = v[i] > w[i] ? v[i] : w[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T min(const vec & v) {                               // Reduce minimum
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp < v[i] ? temp : v[i];
      return temp;
    }
    __host__ __device__ __forceinline__
    friend T max(const vec & v) {                               // Reduce maximum
      T temp = v[0];
      for (int i=1; i<N; i++) temp = temp > v[i] ? temp : v[i];
      return temp;
    }
    __device__ __forceinline__
    friend vec abs(const vec & v) {                             // Absolute value
      vec temp;
      Unroll<Ops::Abs<T>,T,N>::loop(temp,v);
      return temp;
    }
    __device__ __forceinline__
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
      vec temp;
      Unroll<Ops::Rsqrt<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec sin(const vec & v) {                             // Sine function
      vec temp;
      Unroll<Ops::Sin<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend vec cos(const vec & v) {                             // Cosine function
      vec temp;
      Unroll<Ops::Cos<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      Unroll<Ops::SinCos<T>,T,N>::loop(s,c,v);
    }
    __host__ __device__ __forceinline__
    friend vec exp(const vec & v) {                             // Exponential function
      vec temp;
      Unroll<Ops::Exp<T>,T,N>::loop(temp,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    friend int wrap(vec & v, const T & w) {                     // Wrap around periodic boundary
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
    friend void unwrap(vec & v, const T & w, const int & iw) {  // Undo wrap around periodic boundary
      for (int i=0; i<N; i++) {
	if((iw >> i) & 1) v[i] += (v[i] > 0 ? -w : w);
      }
    }
  };
#endif

#if __MIC__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for MIC")
#endif
#include <immintrin.h>
  template<>
  class vec<16,float> {
  private:
    __m512 data;
  public:
    vec(){}                                                     // Default constructor
    vec(const float v) {                                        // Copy constructor scalar
      data = _mm512_set1_ps(v);
    }
    vec(const __m512 v) {                                       // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d,
	const float e, const float f, const float g, const float h,
	const float i, const float j, const float k, const float l,
	const float m, const float n, const float o, const float p) {// Copy constructor (component-wise)
      data = _mm512_setr_ps(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);
    }
    ~vec(){}                                                    // Destructor
    const vec &operator=(const float v) {                       // Scalar assignment
      data = _mm512_set1_ps(v);
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data = _mm512_add_ps(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data = _mm512_sub_ps(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data = _mm512_mul_ps(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data = _mm512_div_ps(data,v.data);
      return *this;
    }
    const vec &operator&=(const __mmask16 & v) {                // Vector compound assignment (and)
      data = _mm512_mask_mov_ps(_mm512_setzero_ps(),v,data);
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(_mm512_add_ps(data,v.data));
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(_mm512_sub_ps(data,v.data));
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(_mm512_mul_ps(data,v.data));
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(_mm512_div_ps(data,v.data));
    }
    __mmask16 operator>(const vec & v) const {                  // Vector arithmetic (greater than)
      return _mm512_cmp_ps_mask(data,v.data,_MM_CMPINT_GT);
    }
    __mmask16 operator<(const vec & v) const {                  // Vector arithmetic (less than)
      return _mm512_cmp_ps_mask(data,v.data,_MM_CMPINT_LT);
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(_mm512_sub_ps(_mm512_setzero_ps(),data));
    }
    float &operator[](int i) {                                  // Indexing (lvalue)
      return ((float*)&data)[i];
    }
    const float &operator[](int i) const {                      // Indexing (rvalue)
      return ((float*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<16; i++) s << v[i] << ' ';
      return s;
    }
    friend float sum(const vec & v) {                           // Sum vector
      return _mm512_reduce_add_ps(v.data);
    }
    friend float norm(const vec & v) {                          // L2 norm squared
      __m512 temp = _mm512_mul_ps(v.data,v.data);
      return _mm512_reduce_add_ps(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(_mm512_min_ps(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(_mm512_max_ps(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec temp = vec(_mm512_rsqrt23_ps(v.data));
      temp *= (temp * temp * v - 3.0f) * (-0.5f);
      return temp;
#else
      return vec(_mm512_rsqrt23_ps(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(_mm512_sin_ps(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(_mm512_cos_ps(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = _mm512_sincos_ps(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(_mm512_exp_ps(v.data));
    }
  };

  template<>
  class vec<8,double> {
  private:
    __m512d data;
  public:
    vec(){}                                                     // Default constructor
    vec(const double v) {                                       // Copy constructor scalar
      data = _mm512_set1_pd(v);
    }
    vec(const __m512d v) {                                      // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const double a, const double b, const double c, const double d,
	const double e, const double f, const double g, const double h) {// Copy constructor (component-wise)
      data = _mm512_setr_pd(a,b,c,d,e,f,g,h);
    }
    ~vec(){}                                                    // Destructor
    const vec &operator=(const double v) {                      // Scalar assignment
      data = _mm512_set1_pd(v);
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data = _mm512_add_pd(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data = _mm512_sub_pd(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data = _mm512_mul_pd(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data = _mm512_div_pd(data,v.data);
      return *this;
    }
    const vec &operator&=(const __mmask8 & v) {                 // Vector compound assignment (and)
      data = _mm512_mask_mov_pd(_mm512_setzero_pd(),v,data);
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(_mm512_add_pd(data,v.data));
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(_mm512_sub_pd(data,v.data));
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(_mm512_mul_pd(data,v.data));
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(_mm512_div_pd(data,v.data));
    }
    __mmask8 operator>(const vec & v) const {                   // Vector arithmetic (greater than)
      return _mm512_cmp_pd_mask(data,v.data,_MM_CMPINT_GT);
    }
    __mmask8 operator<(const vec & v) const {                   // Vector arithmetic (less than)
      return _mm512_cmp_pd_mask(data,v.data,_MM_CMPINT_LT);
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(_mm512_sub_pd(_mm512_setzero_pd(),data));
    }
    double &operator[](int i) {                                 // Indexing (lvalue)
      return ((double*)&data)[i];
    }
    const double &operator[](int i) const {                     // Indexing (rvalue)
      return ((double*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<8; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {                          // Sum vector
      return _mm512_reduce_add_pd(v.data);
    }
    friend double norm(const vec & v) {                         // L2 norm squared
      __m512d temp = _mm512_mul_pd(v.data,v.data);;
      return _mm512_reduce_add_pd(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(_mm512_min_pd(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(_mm512_max_pd(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON
      vec<16,float> in(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],0,0,0,0,0,0,0,0);
      vec<16,float> temp = rsqrt(in);
      temp *= (temp * temp * in - 3.0f) * (-0.5f);
      vec<8,double> out(temp[0],temp[1],temp[2],temp[3],temp[4],temp[5],temp[6],temp[7]);
      return out;
#else
      vec one = 1;
      return vec(_mm512_div_pd(one.data,_mm512_sqrt_pd(v.data)));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(_mm512_sin_pd(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(_mm512_cos_pd(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = _mm512_sincos_pd(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(_mm512_exp_pd(v.data));
    }
  };
#endif

#if __AVX__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for AVX")
#endif
  template<>
  class vec<8,float> {
  private:
    Vec8f data;
  public:
    vec(){}                                                     // Default constructor
    vec(const float v) : data(v) {}                             // Copy constructor scalar
    vec(const Vec8f v) {                                        // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d,
	const float e, const float f, const float g, const float h) :
      data(a,b,c,d,e,f,g,h) {}                                  // Copy constructor (component-wise)
    ~vec(){}                                                    // Destructor
    const vec &operator=(const float v) {                       // Scalar assignment
      data = v;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data += v.data;
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data -= v.data;
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data *= v.data;
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data /= v.data;
      return *this;
    }
    const vec &operator&=(const Vec8fb & v) {                   // Vector compound assignment (bitwise and)
      data = data & v;
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(data + v.data);
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(data - v.data);
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(data * v.data);
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(data / v.data);
    }
    Vec8fb operator>(const vec & v) const {                     // Vector arithmetic (greater than)
      return data > v.data;
    }
    Vec8fb operator<(const vec & v) const {                     // Vector arithmetic (less than)
      return data < v.data;
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(-data);
    }
    float &operator[](int i) {                                  // Indexing (lvalue)
      return ((float*)&data)[i];
    }
    const float &operator[](int i) const {                      // Indexing (rvalue)
      return ((float*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<8; i++) s << v[i] << ' ';
      return s;
    }
    friend float sum(const vec & v) {                           // Sum vector
      return horizontal_add(v.data);
    }
    friend float norm(const vec & v) {                          // L2 norm squared
      Vec8f temp = v.data * v.data;
      return horizontal_add(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(min(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(max(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec temp = vec(approx_rsqrt(v.data));
      temp *= (temp * temp * v - 3.0f) * (-0.5f);
      return temp;
#else
      return vec(approx_rsqrt(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(sin(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(cos(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = sincos(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(exp(v.data));
    }
  };

  template<>
  class vec<4,double> {
  private:
    Vec4d data;
  public:
    vec(){}                                                     // Default constructor
    vec(const double v) : data(v) {}                            // Copy constructor scalar
    vec(const Vec4d v) {                                        // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const double a, const double b, const double c, const double d) :
      data(a,b,c,d) {}                                          // Copy constructor (component-wise)
    ~vec(){}                                                    // Destructor
    const vec &operator=(const double v) {                      // Scalar assignment
      data = v;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data += v.data;
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data -= v.data;
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data *= v.data;
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data /= v.data;
      return *this;
    }
    const vec &operator&=(const Vec4db & v) {                   // Vector compound assignment (bitwise and)
      data = data & v;
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(data + v.data);
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(data - v.data);
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(data * v.data);
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(data / v.data);
    }
    Vec4db operator>(const vec & v) const {                     // Vector arithmetic (greater than)
      return data > v.data;
    }
    Vec4db operator<(const vec & v) const {                     // Vector arithmetic (less than)
      return data < v.data;
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(-data);
    }
    double &operator[](int i) {                                 // Indexing (lvalue)
      return ((double*)&data)[i];
    }
    const double &operator[](int i) const {                     // Indexing (rvalue)
      return ((double*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<4; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {                          // Sum vector
      return horizontal_add(v.data);
    }
    friend double norm(const vec & v) {                         // L2 norm squared
      Vec4d temp = v.data * v.data;
      return horizontal_add(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(min(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(max(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec<8,float> in(v[0],v[1],v[2],v[3],0,0,0,0);
      vec<8,float> temp = rsqrt(in);
      temp *= (temp * temp * in - 3.0f) * (-0.5f);
      vec<4,double> out(temp[0],temp[1],temp[2],temp[3]);
      return out;
#else
      vec one = 1;
      return vec(one.data / sqrt(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(sin(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(cos(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = sincos(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(exp(v.data));
    }
  };
#endif

#if __bgq__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for BG/Q")
#endif
  template<>
  class vec<4,double> {
  private:
    vector4double data;
  public:
    vec(){}                                                     // Default constructor
    vec(const double v) {                                       // Copy constructor scalar
      vector4double temp = {v};
      data = temp;
    }
    vec(const vector4double v) {                                // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const double a, const double b, const double c, const double d) {// Copy constructor (component-wise)
      vector4double temp = {a,b,c,d};
      data = temp;
    }
    ~vec(){}                                                    // Destructor
    const vec &operator=(double v) {                            // Scalar assignment
      vector4double temp = {v};
      data = temp;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data = vec_add(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data = vec_sub(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data = vec_mul(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data = vec_swdiv_nochk(data,v.data);
      return *this;
    }
    const vec &operator&=(const vec & v) {                      // Vector compound assignment (bitwise and)
      data = vec_and(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(vec_add(data,v.data));
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(vec_sub(data,v.data));
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(vec_mul(data,v.data));
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(vec_swdiv_nochk(data,v.data));
    }
    vec operator>(const vec & v) const {                        // Vector arithmetic (greater than)
      return vec(vec_cmpgt(data,v.data));
    }
    vec operator<(const vec & v) const {                        // Vector arithmetic (less than)
      return vec(vec_cmplt(data,v.data));
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(vec_sub((vector4double)(0),data));
    }
    double &operator[](int i) {                                 // Indexing (lvalue)
      return ((double*)&data)[i];
    }
    const double &operator[](int i) const {                     // Indexing (rvalue)
      return ((double*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<4; i++) s << vec_extract(v.data,i) << ' ';
      return s;
    }
    friend double sum(const vec & v) {                          // Sum vector
      double temp = 0;
      for (int i=0; i<4; i++) temp += v[i];
      return temp;
    }
    friend double norm(const vec & v) {                         // L2 norm squared
      double temp = 0;
      for (int i=0; i<4; i++) temp += v[i] * v[i];
      return temp;
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      vec temp;
      for (int i=0; i<4; i++) temp[i] = v[i] < w[i] ? v[i] : w[i];
      return temp;
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      vec temp;
      for (int i=0; i<4; i++) temp[i] = v[i] > w[i] ? v[i] : w[i];
      return temp;
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec temp = vec(vec_rsqrtes(v.data));
      temp *= (temp * temp * v - 3.0f) * (-0.5f);
      return temp;
#else
      return vec(vec_rsqrtes(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(sind4(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(cosd4(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      sincosd4(v.data, s.data, c.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(expd4(v.data));
    }
  };
#endif

#if __SSE__
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for SSE")
#endif
  template<>
  class vec<4,float> {
  private:
    Vec4f data;
  public:
    vec(){}                                                     // Default constructor
    vec(const float v) : data(v) {}                             // Copy constructor scalar
    vec(const Vec4f v) {                                        // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const float a, const float b, const float c, const float d) :
      data(a,b,c,d) {}                                          // Copy constructor (component-wise)
    ~vec(){}                                                    // Destructor
    const vec &operator=(const float v) {                       // Scalar assignment
      data = v;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data += v.data;
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data -= v.data;
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data *= v.data;
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data /= v.data;
      return *this;
    }
    const vec &operator&=(const Vec4fb & v) {                   // Vector compound assignment (bitwise and)
      data = data & v;
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(data + v.data);
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(data - v.data);
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(data * v.data);
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(data / v.data);
    }
    Vec4fb operator>(const vec & v) const {                     // Vector arithmetic (greater than)
      return data > v.data;
    }
    Vec4fb operator<(const vec & v) const {                     // Vector arithmetic (less than)
      return data < v.data;
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(-data);
    }
    float &operator[](int i) {                                  // Indexing (lvalue)
      return ((float*)&data)[i];
    }
    const float &operator[](int i) const {                      // Indexing (rvalue)
      return ((float*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<4; i++) s << v[i] << ' ';
      return s;
    }
    friend float sum(const vec & v) {                           // Sum vector
      return horizontal_add(v.data);
    }
    friend float norm(const vec & v) {                          // L2 norm squared
      Vec4f temp = v.data * v.data;
      return horizontal_add(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(min(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(max(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec temp = vec(approx_rsqrt(v.data));
      temp *= (temp * temp * v - 3.0f) * (-0.5f);
      return temp;
#else
      return vec(approx_rsqrt(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(sin(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(cos(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = sincos(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(exp(v.data));
    }
  };

  template<>
  class vec<2,double> {
  private:
    Vec2d data;
  public:
    vec(){}                                                     // Default constructor
    vec(const double v) : data(v) {}                            // Copy constructor scalar
    vec(const Vec2d v) {                                        // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const double a, const double b) :
      data(a,b) {}                                              // Copy constructor (component-wise)
    ~vec(){}                                                    // Destructor
    const vec &operator=(const double v) {                      // Scalar assignment
      data = v;
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data += v.data;
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data -= v.data;
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data *= v.data;
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data /= v.data;
      return *this;
    }
    const vec &operator&=(const Vec2db & v) {                   // Vector compound assignment (bitwise and)
      data = data & v;
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(data + v.data);
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(data - v.data);
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(data * v.data);
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(data / v.data);
    }
    Vec2db operator>(const vec & v) const {                     // Vector arithmetic (greater than)
      return data > v.data;
    }
    Vec2db operator<(const vec & v) const {                     // Vector arithmetic (less than)
      return data < v.data;
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(-data);
    }
    double &operator[](int i) {                                 // Indexing (lvalue)
      return ((double*)&data)[i];
    }
    const double &operator[](int i) const {                     // Indexing (rvalue)
      return ((double*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<2; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {                          // Sum vector
      return horizontal_add(v.data);
    }
    friend double norm(const vec & v) {                         // L2 norm squared
      Vec2d temp = v.data * v.data;
      return horizontal_add(temp);
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(min(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(max(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec<4,float> in(v[0],v[1],0,0);
      vec<4,float> temp = rsqrt(in);
      temp *= (temp * temp * in - 3.0f) * (-0.5f);
      vec<2,double> out(temp[0],temp[1]);
      return out;
#else
      vec one = 1;
      return vec(one.data / sqrt(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      return vec(sin(v.data));
    }
    friend vec cos(const vec & v) {                             // Cosine function
      return vec(cos(v.data));
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s.data = sincos(&c.data, v.data);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      return vec(exp(v.data));
    }
  };
#endif

#if __sparc_v9__ & EXAFMM_USE_SIMD
#if EXAFMM_VEC_VERBOSE
#pragma message("Overloading vector operators for SPARC")
#endif
#include <emmintrin.h>

  template<>
  class vec<2,double> {
  private:
    __m128d data;
  public:
    vec(){}                                                     // Default constructor
    vec(const double v) {                                       // Copy constructor scalar
      data = _mm_set_pd(v,v);
    }
    vec(const __m128d v) {                                      // Copy constructor SIMD register
      data = v;
    }
    vec(const vec & v) {                                        // Copy constructor vector
      data = v.data;
    }
    vec(const double a, const double b) {                       // Copy constructor (component-wise)
      data = _mm_set_pd(b,a);
    }
    ~vec(){}                                                    // Destructor
    const vec &operator=(const double v) {                      // Scalar assignment
      data = _mm_set_pd(v,v);
      return *this;
    }
    const vec &operator=(const vec & v) {                       // Vector assignment
      data = v.data;
      return *this;
    }
    const vec &operator+=(const vec & v) {                      // Vector compound assignment (add)
      data = _mm_add_pd(data,v.data);
      return *this;
    }
    const vec &operator-=(const vec & v) {                      // Vector compound assignment (subtract)
      data = _mm_sub_pd(data,v.data);
      return *this;
    }
    const vec &operator*=(const vec & v) {                      // Vector compound assignment (multiply)
      data = _mm_mul_pd(data,v.data);
      return *this;
    }
    const vec &operator/=(const vec & v) {                      // Vector compound assignment (divide)
      data = _mm_mul_pd(data,_fjsp_rcpa_v2r8(v.data));
      return *this;
    }
    const vec &operator&=(const vec & v) {                      // Vector compound assignment (bitwise and)
      data = _mm_and_pd(data,v.data);
      return *this;
    }
    vec operator+(const vec & v) const {                        // Vector arithmetic (add)
      return vec(_mm_add_pd(data,v.data));
    }
    vec operator-(const vec & v) const {                        // Vector arithmetic (subtract)
      return vec(_mm_sub_pd(data,v.data));
    }
    vec operator*(const vec & v) const {                        // Vector arithmetic (multiply)
      return vec(_mm_mul_pd(data,v.data));
    }
    vec operator/(const vec & v) const {                        // Vector arithmetic (divide)
      return vec(_mm_mul_pd(data,_fjsp_rcpa_v2r8(v.data)));
    }
    vec operator>(const vec & v) const {                        // Vector arithmetic (greater than)
      return vec(_mm_cmpgt_pd(data,v.data));
    }
    vec operator<(const vec & v) const {                        // Vector arithmetic (less than)
      return vec(_mm_cmplt_pd(data,v.data));
    }
    vec operator-() const {                                     // Vector arithmetic (negation)
      return vec(_mm_sub_pd(_mm_setzero_pd(),data));
    }
    double &operator[](int i) {                                 // Indexing (lvalue)
      return ((double*)&data)[i];
    }
    const double &operator[](int i) const {                     // Indexing (rvalue)
      return ((double*)&data)[i];
    }
    friend std::ostream &operator<<(std::ostream & s, const vec & v) {// Component-wise output stream
      for (int i=0; i<2; i++) s << v[i] << ' ';
      return s;
    }
    friend double sum(const vec & v) {                          // Sum vector
      return v[0] + v[1];
    }
    friend double norm(const vec & v) {                         // L2 norm squared
      return v[0] * v[0] + v[1] * v[1];
    }
    friend vec min(const vec & v, const vec & w) {              // Element-wise minimum
      return vec(_mm_min_pd(v.data,w.data));
    }
    friend vec max(const vec & v, const vec & w) {              // Element-wise maximum
      return vec(_mm_max_pd(v.data,w.data));
    }
    friend vec rsqrt(const vec & v) {                           // Reciprocal square root
#if EXAFMM_VEC_NEWTON                                           // Switch on Newton-Raphson correction
      vec temp = vec(_fjsp_rsqrta_v2r8(v.data));
      temp *= (temp * temp * v - 3.0f) * (-0.5f);
      return temp;
#else
      return vec(_fjsp_rsqrta_v2r8(v.data));
#endif
    }
    friend vec sin(const vec & v) {                             // Sine function
      vec temp;
      temp[0] = std::sin(v[0]);
      temp[1] = std::sin(v[1]);
      return temp;
    }
    friend vec cos(const vec & v) {                             // Cosine function
      vec temp;
      temp[0] = std::cos(v[0]);
      temp[1] = std::cos(v[1]);
      return temp;
    }
    friend void sincos(vec & s, vec & c, const vec & v) {       // Sine & cosine function
      s[0] = std::sin(v[0]);
      s[1] = std::sin(v[1]);
      c[0] = std::cos(v[0]);
      c[1] = std::cos(v[1]);
    }
    friend vec exp(const vec & v) {                             // Exponential function
      vec temp;
      temp[0] = std::exp(v[0]);
      temp[1] = std::exp(v[2]);
      return temp;
    }
  };
#endif

}
#endif
