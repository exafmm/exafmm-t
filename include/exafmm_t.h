#ifndef exafmm_t_h
#define exafmm_t_h
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>
#include <omp.h>
#include <set>
#include <vector>
#include "align.h"
#include "args.h"
#include "vec.h"

namespace exafmm_t {
  const int MEM_ALIGN = 64;
  const int CACHE_SIZE = 512;
  const int NCHILD = 8;

#if FLOAT
  typedef float real_t;                       //!< Real number type
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
  typedef double real_t;                       //!< Real number type
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

  const real_t PI = M_PI;
  typedef std::complex<real_t> complex_t;       //!< Complex number type
  typedef vec<3,int> ivec3;                     //!< Vector of 3 int types
  typedef vec<3,real_t> vec3;                   //!< Vector of 3 real_t types
  typedef vec<3,complex_t> cvec3;               //!< Vector of 3 complex_t types

  //! SIMD vector types for AVX512, AVX, and SSE
  const int NSIMD = SIMD_BYTES / int(sizeof(real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
  typedef vec<NSIMD, real_t> simdvec;           //!< SIMD vector of NSIMD real_t types

  typedef std::vector<real_t> RealVec;          //!< Vector of real_t types
  typedef std::vector<complex_t> ComplexVec;    //!< Vector of complex_t types
  typedef AlignedAllocator<real_t, MEM_ALIGN> AlignAllocator;   //!< Allocator for memory alignment
  typedef std::vector<real_t, AlignAllocator> AlignedVec;       //!< Aligned vector of real_t types

  //! Interaction Type
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

  //! Structure of bodies.
  template <typename T>
  struct Body {
#if SORT_BACK
    int ibody;                                  //!< Initial body numbering for sorting back
#endif
    vec3 X;                                     //!< Coordinates
    T q;                                   //!< Charge
    T p;                                   //!< Potential
    vec<3,T> F;                            //!< Force
  };
  template <typename T> using Bodies = std::vector<Body<T>>;              //!< Vector of nodes

  //! Structure of nodes.
  template <typename T>
  struct Node {
    size_t idx;                                 //!< Index in the octree
    size_t idx_M2L;                             //!< Index in global M2L interaction list
    bool is_leaf;                               //!< Whether the node is leaf
    int ntrgs;                                  //!< Number of targets
    int nsrcs;                                  //!< Number of sources
    vec3 x;                                     //!< Coordinates of the center of the node
    real_t r;                                   //!< Radius of the node
    uint64_t key;                               //!< Morton key
    int level;                                  //!< Level in the octree
    int octant;                                 //!< Octant
    Node* parent;                               //!< Pointer to parent
    std::vector<Node*> children;                //!< Vector of pointers to child nodes
    std::vector<Node*> colleagues;              //!< Vector of pointers to colleague nodes
    std::vector<Node*> P2L_list;                //!< Vector of pointers to nodes in P2L interaction list
    std::vector<Node*> M2P_list;                //!< Vector of pointers to nodes in M2P interaction list
    std::vector<Node*> P2P_list;                //!< Vector of pointers to nodes in P2P interaction list
    std::vector<Node*> M2L_list;                //!< Vector of pointers to nodes in M2L interaction list
#if SORT_BACK
    std::vector<int> isrcs;                     //!< Vector of initial source numbering
    std::vector<int> itrgs;                     //!< Vector of initial target numbering
#endif
    RealVec src_coord;                          //!< Vector of coordinates of sources in the node
    RealVec trg_coord;                          //!< Vector of coordinates of targets in the node
    std::vector<T> src_value;                   //!< Vector of charges of sources in the node
    std::vector<T> trg_value;                   //!< Vector of potentials and gradients of targets in the node
    std::vector<T> up_equiv;                    //!< Upward check potentials / Upward equivalent densities
    std::vector<T> dn_equiv;                    //!< Downward check potentials / Downward equivalent densites
  };
  
  // alias template
  template <typename T> using Nodes = std::vector<Node<T>>;              //!< Vector of nodes
  template <typename T> using NodePtrs = std::vector<Node<T>*>;          //!< Vector of Node pointers
  using Keys = std::vector<std::set<uint64_t>>; //!< Vector of Morton keys of each level

  //! 
  struct M2LData {
    std::vector<size_t> fft_offset;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_offset;  // target's first child's dnward_equiv's displacement
    RealVec ifft_scale;
    std::vector<size_t> interaction_offset_f;
    std::vector<size_t> interaction_count_offset;
  };

  // Relative coordinates and interaction lists
  extern std::vector<std::vector<ivec3>> REL_COORD;  //!< Vector of possible relative coordinates (inner) of each interaction type (outer)
  extern std::vector<std::vector<int>> HASH_LUT;     //!< Vector of hash Lookup tables (inner) of relative positions for each interaction type (outer)

  // Precomputation matrices
#if HELMHOLTZ
  extern std::vector<ComplexVec> matrix_UC2E_U, matrix_UC2E_V;
  extern std::vector<ComplexVec> matrix_DC2E_U, matrix_DC2E_V;
  extern std::vector<std::vector<ComplexVec>> matrix_M2M, matrix_L2L;
#else
  /*
  extern RealVec matrix_UC2E_U;       //!< Upward check to upward equivalent precomputation matrix, first component
  extern RealVec matrix_UC2E_V;       //!< Upward check to upward equivalent precomputation matrix, second component
  extern RealVec matrix_DC2E_U;       //!< Downward check to downward equivalent precomputation matrix, first component
  extern RealVec matrix_DC2E_V;       //!< Downward check to downward equivalent precomputation matrix, second component
  extern std::vector<RealVec> matrix_M2M;     //!< M2M precomputation matrix
  extern std::vector<RealVec> matrix_L2L;     //!< L2L precomputation matrix
  extern std::vector<AlignedVec> matrix_M2L;  //!< M2L precomputation matrix
  */
#endif

  extern int P;                               //!< Order of multipole expansion (number of points along each edge of the equivalent/check surface)
  extern int NSURF;                           //!< Number of points on equivalent/check surface
  extern int MAXLEVEL;                        //!< Max level of the octree
  extern vec3 X0;                             //!< Coordinates of the center of root node
  extern real_t R0;                           //!< Radius of root node
#if HELMHOLTZ
  extern real_t WAVEK;                        //!< Wavenumber only for Helmholtz kernel
#endif
}
#endif
