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
#include "mpi_utils.h"

namespace exafmm_t {
  const int MEM_ALIGN = 64;
  const int CACHE_SIZE = 512;
  const int NCHILD = 8;

#if FLOAT
  typedef float real_t;                       //!< Real number type
  MPI_Datatype MPI_REAL_T = MPI_FLOAT;        //!< Floating point MPI type
  const real_t EPS = 1e-8f;
  typedef fftwf_complex fft_complex;
  typedef fftwf_plan fft_plan;
#define fft_plan_dft fftwf_plan_dft
#define fft_plan_many_dft fftwf_plan_many_dft
#define fft_execute_dft fftwf_execute_dft
#define fft_plan_dft_r2c fftwf_plan_dft_r2c
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#define fft_flops fftwf_flops
#else
  typedef double real_t;                       //!< Real number type
  MPI_Datatype MPI_REAL_T = MPI_DOUBLE;        //!< Floating point MPI type
  const real_t EPS = 1e-16;
  typedef fftw_complex fft_complex;
  typedef fftw_plan fft_plan;
#define fft_plan_dft fftw_plan_dft
#define fft_plan_many_dft fftw_plan_many_dft
#define fft_execute_dft fftw_execute_dft
#define fft_plan_dft_r2c fftw_plan_dft_r2c
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#define fft_flops fftw_flops
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

  //! Interaction types that need to be pre-computed.
  typedef enum {
    M2M_Type = 0,
    L2L_Type = 1,
    M2L_Helper_Type = 2,
    M2L_Type = 3,
    Type_Count = 4
  } Precompute_Type;

  /**
   * @brief Structure of bodies.
   * 
   * @tparam T Value type of sources and targets (real or complex).
   */
  template <typename T>
  struct Body {
    int ibody;                             //!< Initial body numbering for sorting back
    vec3 X;                                //!< Coordinates
    uint64_t key;                          //!< Hilbert key
    T q;                                   //!< Charge
    T p;                                   //!< Potential
    vec<3,T> F;                            //!< Gradient
  };
  template <typename T> using Bodies = std::vector<Body<T>>;     //!< Vector of nodes

  /**
   * @brief Base structure of nodes, used for MPI communications.
   */
  struct NodeBase {
    vec3 x;               //!< Coordinates of the center of the node
    real_t r;             //!< Radius of the node
    uint64_t key;         //!< Hilbert key
    bool is_leaf;         //!< Whether the node is leaf
    int nsrcs;            //!< Number of sources
  };

  /**
   * @brief Structure of nodes.
   * 
   * @tparam Value type of sources and targets (real or complex).
   */
  template <typename T>
  struct Node : public NodeBase {
    size_t idx;                                 //!< Index in the octree
    size_t idx_M2L;                             //!< Index in global M2L interaction list
    int ntrgs;                                  //!< Number of targets
    int nsrcs;                                  //!< Number of sources
    int level;                                  //!< Level in the octree
    int octant;                                 //!< Octant
    Node* parent;                               //!< Pointer to parent
    std::vector<Node*> children;                //!< Vector of pointers to child nodes
    std::vector<Node*> P2L_list;                //!< Vector of pointers to nodes in P2L interaction list
    std::vector<Node*> M2P_list;                //!< Vector of pointers to nodes in M2P interaction list
    std::vector<Node*> P2P_list;                //!< Vector of pointers to nodes in P2P interaction list
    std::vector<Node*> M2L_list;                //!< Vector of pointers to nodes in M2L interaction list
    std::vector<int> isrcs;                     //!< Vector of initial source numbering
    std::vector<int> itrgs;                     //!< Vector of initial target numbering
    RealVec src_coord;                          //!< Vector of coordinates of sources in the node
    RealVec trg_coord;                          //!< Vector of coordinates of targets in the node
    std::vector<T> src_value;                   //!< Vector of charges of sources in the node
    std::vector<T> trg_value;                   //!< Vector of potentials and gradients of targets in the node
    std::vector<T> up_equiv;                    //!< Upward check potentials / Upward equivalent densities
    std::vector<T> dn_equiv;                    //!< Downward check potentials / Downward equivalent densites
  };
  
  // alias template
  template <typename T> using Nodes = std::vector<Node<T>>;        //!< Vector of nodes
  template <typename T> using NodePtrs = std::vector<Node<T>*>;    //!< Vector of Node pointers
  using Keys = std::vector<std::set<uint64_t>>;                    //!< Vector of Morton keys of each level

  //! M2L setup data
  struct M2LData {
    std::vector<size_t> fft_offset;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_offset;  // target's first child's dnward_equiv's displacement
    RealVec ifft_scale;
    std::vector<size_t> interaction_offset_f;
    std::vector<size_t> interaction_count_offset;
  };

  // Relative coordinates and interaction lists
  std::vector<std::vector<ivec3>> REL_COORD;  //!< Vector of possible relative coordinates (inner) of each interaction type (outer)
  std::vector<std::vector<int>> HASH_LUT;     //!< Vector of hash Lookup tables (inner) of relative positions for each interaction type (outer)
  std::vector<std::vector<int>> M2L_INDEX_MAP;  //!< [M2L_relpos_idx][octant] -> M2L_Helper_relpos_idx
}
#endif
