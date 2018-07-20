#ifndef exafmm_t_h
#define exafmm_t_h
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fftw3.h>
#include <vector>
#include "align.h"
#include "profile.h"
#include "vec.h"

extern "C" {
  void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA, float* A,
              int* LDA, float* B, int* LDB, float* BETA, float* C, int* LDC);
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
              int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  void sgesvd_(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
               float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO);
  void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
               double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
}

namespace exafmm_t {
#define MEM_ALIGN 64
#define CACHE_SIZE 512

#if FLOAT
  typedef float real_t;
  const real_t EPS = 1e-8f;
  typedef fftwf_complex fft_complex;
  typedef fftwf_plan fft_plan;
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

  //! using blas gemm with row major data
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C) {
    char transA = 'N', transB = 'N';
    real_t alpha = 1.0, beta = 0.0;
#if FLOAT
    sgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#else
    dgemm_(&transA, &transB, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
#endif
  }

  //! lapack svd with row major data: A = U*S*VT, A is m by n
  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT) {
    char JOBU = 'S', JOBVT = 'S';
    int INFO;
    int LWORK = std::max(3*std::min(m,n)+std::max(m,n), 5*std::min(m,n));
    LWORK = std::max(LWORK, 1);
    int k = std::min(m, n);
    RealVec tS(k, 0.);
    RealVec WORK(LWORK);
#if FLOAT
    sgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#else
    dgesvd_(&JOBU, &JOBVT, &n, &m, A, &n, &tS[0], VT, &n, U, &k, &WORK[0], &LWORK, &INFO);
#endif
    // copy singular values from 1d layout (tS) to 2d layout (S)
    for(int i=0; i<k; i++) {
      S[i*n+i] = tS[i];
    }
  }

  RealVec transpose(RealVec& vec, int m, int n) {
    RealVec temp(vec.size());
    for(int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
        temp[j*m+i] = vec[i*n+j];
      }
    }
    return temp;
  }

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
    real_t q;                                   //!< Charge
    real_t p;                                   //!< Potential
    vec3 F;                                     //!< Force
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

    size_t idx;
    size_t node_id;
    int depth;
    int octant;
    real_t coord[3];
    Node* parent;
    std::vector<Node*> child;
    Node* colleague[27];
    std::vector<Node*> interac_list[Type_Count];
    RealVec pt_coord;
    RealVec pt_src;  // src's charge
    RealVec pt_trg;  // trg's potential
    RealVec upward_equiv; // M
    RealVec dnward_equiv; // L

    bool IsLeaf() {
      return numChilds == 0;
    }

    Node* Child(int id) {
      return (numChilds == 0) ? NULL : child[id];
    }
  };
  typedef std::vector<Node> Nodes;              //!< Vector of nodes

  struct M2LData {
    std::vector<size_t> fft_vec;   // source's first child's upward_equiv's displacement
    std::vector<size_t> ifft_vec;  // target's first child's dnward_equiv's displacement
    RealVec fft_scl;
    RealVec ifft_scl;
    std::vector<size_t> interac_vec;
    std::vector<size_t> interac_dsp;
  };
  M2LData M2Ldata;

  // Relative coordinates and interaction lists
  std::vector<std::vector<ivec3>> rel_coord;
  std::vector<std::vector<int>> hash_lut;     // coord_hash -> index in rel_coord

  // Precomputation matrices
  RealVec M2M_U, M2M_V;
  RealVec L2L_U, L2L_V;
  std::vector<RealVec> mat_M2M, mat_L2L;
  std::vector<RealVec> mat_M2L;
  std::vector<RealVec> mat_M2L_Helper;

  int MULTIPOLE_ORDER;   // order of multipole expansion
  int NSURF;     // number of surface coordinates
  int NCRIT;
  int MAXLEVEL;
  const int NCHILD = 8;
}
#endif
