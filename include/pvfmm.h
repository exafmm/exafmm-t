#ifndef pvfmm_h
#define pvfmm_h
#include "align.h"
#include "matrix.hpp"
#include "vec.h"

namespace pvfmm {
#ifndef NULL
#define NULL 0
#endif

#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX_DEPTH 62
#define MEM_ALIGN 64

#if FLOAT
  typedef float real_t;
  typedef fftwf_complex fft_complex;
  typedef fftwf_plan fft_plan;
#define fft_plan_many_dft_r2c fftwf_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftwf_plan_many_dft_c2r
#define fft_execute_dft_r2c fftwf_execute_dft_r2c
#define fft_execute_dft_c2r fftwf_execute_dft_c2r
#define fft_destroy_plan fftwf_destroy_plan
#else
  typedef double real_t;
  typedef fftw_complex fft_complex;
  typedef fftw_plan fft_plan;
#define fft_plan_many_dft_r2c fftw_plan_many_dft_r2c
#define fft_plan_many_dft_c2r fftw_plan_many_dft_c2r
#define fft_execute_dft_r2c fftw_execute_dft_r2c
#define fft_execute_dft_c2r fftw_execute_dft_c2r
#define fft_destroy_plan fftw_destroy_plan
#endif

  typedef vec<3,int> ivec3;                    //!< std::vector of 3 int types
  typedef vec<3,real_t> vec3;                   //!< Vector of 3 real_t types

  //! SIMD vector types for AVX512, AVX, and SSE
  const int NSIMD = SIMD_BYTES / int(sizeof(real_t));  //!< SIMD vector length (SIMD_BYTES defined in vec.h)
  typedef vec<NSIMD, real_t> simdvec;                  //!< SIMD vector type
  typedef AlignedAllocator<real_t, MEM_ALIGN> AlignAllocator;
  typedef std::vector<real_t> RealVec;
  typedef std::vector<real_t, AlignAllocator> AlignedVec;

  typedef enum {
    M2M_V_Type = 0,
    M2M_U_Type = 1,
    L2L_V_Type = 2,
    L2L_U_Type = 3,
    M2M_Type = 4,
    L2L_Type = 5,
    M2L_Helper_Type = 6,
    M2L_Type = 7,
    P2P0_Type = 10,
    P2P1_Type = 11,
    P2P2_Type = 12,
    M2P_Type = 13,
    P2L_Type = 14,
    Type_Count = 15
  } Mat_Type;

  typedef enum {
    Scaling = 0,
    ReflecX = 1,
    ReflecY = 2,
    ReflecZ = 3,
    SwapXY  = 4,
    SwapXZ  = 5,
    C_Perm = 6,
    Perm_Count = 12
  } Perm_Type;

  //! Structure of bodies
  struct Body {
    vec3 X;                                     //!< Position
    real_t q;                                   //!< Charge
    real_t p;                                   //!< Potential
    vec3 F;                                     //!< Force
  };
  typedef std::vector<Body> Bodies;             //!< Vector of bodies

  //! Structure of cells
  struct FMM_Node {
    int numChilds;
    int numBodies;
    FMM_Node * fchild;
    Body * body;
    vec3 X;
    real_t R;
    
    size_t idx;
    size_t node_id;
    int depth;
    int octant;
    real_t coord[3];
    FMM_Node* parent;
    std::vector<FMM_Node*> child;
    FMM_Node* colleague[27];
    std::vector<FMM_Node*> interac_list[Type_Count];
    std::vector<real_t> pt_coord;
    std::vector<real_t> pt_src;  // src's charge
    std::vector<real_t> pt_trg;  // trg's potential
    std::vector<real_t> upward_equiv; // M
    std::vector<real_t> dnward_equiv; // L

    bool IsLeaf() {
      return numChilds == 0;
    }

    FMM_Node* Child(int id) {
      return (numChilds == 0) ? NULL : child[id]; 
    }
  };
  typedef std::vector<FMM_Node> FMM_Nodes;              //!< Vector of cells
  FMM_Node* root_node;

  struct M2LData {
    std::vector<size_t> fft_vec;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_vec;  // target's first child's dnward_equiv's displacement
    std::vector<real_t> fft_scl;
    std::vector<real_t> ifft_scl;
    std::vector<size_t> interac_vec;
    std::vector<size_t> interac_dsp;
  };
  M2LData M2Ldata;

  std::vector<std::vector<real_t> > upwd_check_surf;
  std::vector<std::vector<real_t> > upwd_equiv_surf;
  std::vector<std::vector<real_t> > dnwd_check_surf;
  std::vector<std::vector<real_t> > dnwd_equiv_surf;

  std::vector<FMM_Node*> leafs, nonleafs, allnodes;

  // Relative coordinates and interaction lists
  std::vector<std::vector<ivec3> > rel_coord;
  std::vector<std::vector<int> > hash_lut;     // coord_hash -> index in rel_coord
  std::vector<std::vector<int> > interac_class;  // index -> index of abs_coord of the same class
  std::vector<std::vector<std::vector<Perm_Type> > > perm_list;// index -> list of permutations needed in order to change from abs_coord to rel_coord

  // Precomputation matrices and permutations
  Matrix<real_t> M2M_U, M2M_V;
  Matrix<real_t> L2L_U, L2L_V;
  Matrix<real_t> mat_M2M, mat_L2L;
  std::vector<Matrix<real_t> > mat_M2L;
  std::vector<real_t*> mat_M2L_Helper; 
  std::vector<Permutation<real_t> > perm_M2M;
  std::vector<Permutation<real_t> > perm_r, perm_c;

  int LEVEL;     // depth of octree
  int MULTIPOLE_ORDER;   // order of multipole expansion
  int NSURF;     // number of surface coordinates
  int NCRIT;
  int SRC_DIM;    // dimension of source's charge
  int TRG_DIM;    // dimension of target's value (potential + force)
  int POT_DIM;    // dimension of target's potential
  int N1, N2, N3, N3_, FFTSIZE;
}
#endif
