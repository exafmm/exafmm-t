#ifndef exafmm_t_h
#define exafmm_t_h
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <set>
#include <vector>
#include "align.h"
#include "args.h"
#include "vec.h"

namespace exafmm_t {
#define MEM_ALIGN 64
#define CACHE_SIZE 512

#if FLOAT
  typedef float real_t;
  const real_t EPS = 1e-8f;
  typedef fftwf_complex fft_complex;
  typedef fftwf_plan fft_plan;
#define fft_plan_dft fftwf_plan_dft
#define fft_plan_many_dft fftwf_plan_many_dft
#define fft_plan_dft_r2c fftwf_plan_dft_r2c
#define fft_plan_dft_c2r fftwf_plan_dft_c2r
#define fft_execute_dft fftwf_execute_dft
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#else
  typedef double real_t;
  const real_t EPS = 1e-16;
  typedef fftw_complex fft_complex;
  typedef fftw_plan fft_plan;
#define fft_plan_dft fftw_plan_dft
#define fft_plan_many_dft fftw_plan_many_dft
#define fft_plan_dft_r2c fftw_plan_dft_r2c
#define fft_plan_dft_c2r fftw_plan_dft_c2r
#define fft_execute_dft fftw_execute_dft
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#endif

  typedef std::complex<real_t> complex_t;
  typedef vec<3,int> ivec3;                     //!< std::vector of 3 int types
  typedef vec<3,real_t> vec3;                   //!< std::vector of 3 real_t types
  typedef vec<3,complex_t> cvec3;               //!< std::vector of 3 complex_t types

  //! SIMD vector types for AVX512, AVX, and SSE
  const int NSIMD = SIMD_BYTES / int(sizeof(real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
  typedef vec<NSIMD, real_t> simdvec;                  //!< SIMD vector type
  typedef std::vector<real_t> RealVec;
  typedef std::vector<complex_t> ComplexVec;
  typedef AlignedAllocator<real_t, MEM_ALIGN> AlignAllocator;
  typedef std::vector<real_t, AlignAllocator> AlignedVec;

  typedef enum {
    M2M_Type = 0,
    L2L_Type = 1,
    M2L_Helper_Type = 2,
    M2L_Type = 3,
    P2P0_Type = 4,
    P2P1_Type = 5,
    P2P2_Type = 6,
    M2P_Type = 7,
    P2L_Type = 8,
    Type_Count = 9
  } Mat_Type;

  //! Structure of bodies
  struct Body {
    vec3 X;                                     //!< Position
#if COMPLEX
    complex_t q;                                   //!< Charge
    complex_t p;                                   //!< Potential
    cvec3 F;                                     //!< Force
#else
    real_t q;                                   //!< Charge
    real_t p;                                   //!< Potential
    vec3 F;                                     //!< Force
#endif
  };
  typedef std::vector<Body> Bodies;             //!< Vector of bodies

  //! Structure of nodes
  struct Node {
    int numChilds;
    int numBodies;
    Node * fchild;
    Body * body;
    vec3 X;
    real_t R;
    uint64_t key;

    size_t idx;
    size_t node_id;
    int depth;
    int octant;
    real_t coord[3];
    Node* parent;
    std::vector<Node*> child;
    Node* colleague[27];
    std::vector<Node*> P2Llist;
    std::vector<Node*> M2Plist;
    std::vector<Node*> P2Plist;
    std::vector<Node*> M2Llist;
    std::vector<int> M2LRelPos;
    RealVec pt_coord;
    AlignedVec upEquiv;  // upward_equiv in frequency domain
#if COMPLEX
    ComplexVec pt_src;  // src's charge
    ComplexVec pt_trg;  // trg's potential
    ComplexVec upward_equiv; // M
    ComplexVec dnward_equiv; // L
#else
    RealVec pt_src;  // src's charge
    RealVec pt_trg;  // trg's potential
    RealVec upward_equiv; // M
    RealVec dnward_equiv; // L
#endif

    bool IsLeaf() {
      return numChilds == 0;
    }

    Node* Child(int id) {
      return (numChilds == 0) ? NULL : child[id];
    }
  };
  typedef std::vector<Node> Nodes;              //!< Vector of nodes
  typedef std::vector<std::set<uint64_t>> Keys; //!< Vector of Morton keys of each level

  struct M2LData {
    std::vector<size_t> fft_vec;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_vec;  // target's first child's dnward_equiv's displacement
    RealVec fft_scl;
    RealVec ifft_scl;
    std::vector<size_t> interac_vec;
    std::vector<size_t> interac_dsp;
  };

  // Relative coordinates and interaction lists
  extern std::vector<std::vector<ivec3>> rel_coord;

  // Precomputation matrices
  extern RealVec M2M_U, M2M_V;
  extern RealVec L2L_U, L2L_V;
  extern std::vector<RealVec> mat_M2M, mat_L2L, mat_M2L_Helper;

  extern int MULTIPOLE_ORDER;   // order of multipole expansion
  extern int NSURF;     // number of surface coordinates
  extern int MAXLEVEL;
  const int NCHILD = 8;
}
#endif
