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
  typedef vec<NSIMD, real_t> simdvec;           //!< SIMD vector type

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
    size_t idx;
    size_t idx_M2L;
    bool is_leaf;
    int ntrgs;
    int nsrcs;
    vec3 xmin;    // the coordinates of the front-left-bottom corner
    real_t r;
    uint64_t key;
    int level;
    int octant;
    Node* parent;
    std::vector<Node*> children;
    std::vector<Node*> colleagues;
    std::vector<Node*> P2L_list;
    std::vector<Node*> M2P_list;
    std::vector<Node*> P2P_list;
    std::vector<Node*> M2L_list;
    
    RealVec src_coord;
    RealVec trg_coord;
#if COMPLEX
    ComplexVec src_value; // source's charge
    ComplexVec trg_value; // target's potential and gradient
    ComplexVec up_equiv; // M
    ComplexVec dn_equiv; // L
#else
    RealVec src_value; // source's charge
    RealVec trg_value; // target's potential and gradient
    RealVec up_equiv; // M
    RealVec dn_equiv; // L
#endif
  };
  typedef std::vector<Node> Nodes;              //!< Vector of nodes
  typedef std::vector<Node*> NodePtrs;          //!< Vector of Node pointers
  typedef std::vector<std::set<uint64_t>> Keys; //!< Vector of Morton keys of each level

  struct M2LData {
    std::vector<size_t> fft_offset;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_offset;  // target's first child's dnward_equiv's displacement
    RealVec ifft_scale;
    std::vector<size_t> interaction_offset_f;
    std::vector<size_t> interaction_count_offset;
  };

  // Relative coordinates and interaction lists
  extern std::vector<std::vector<ivec3>> rel_coord;

  // Precomputation matrices
#if HELMHOLTZ
  extern std::vector<ComplexVec> UC2E_U, UC2E_V;
  extern std::vector<ComplexVec> DC2E_U, DC2E_V;
  extern std::vector<std::vector<ComplexVec>> matrix_M2M, matrix_L2L;
  extern std::vector<std::vector<RealVec>> matrix_M2L;
#else
  extern RealVec UC2E_U, UC2E_V;
  extern RealVec DC2E_U, DC2E_V;
  extern std::vector<RealVec> matrix_M2M, matrix_L2L, matrix_M2L;
#endif

  extern int MULTIPOLE_ORDER;   // order of multipole expansion
  extern int NSURF;     // number of surface coordinates
  extern int MAXLEVEL;
  extern vec3 XMIN0;    // coordinates of root
  extern real_t R0;     // radius of root
  const int NCHILD = 8;
#if HELMHOLTZ
  extern real_t MU;
#endif
}
#endif
