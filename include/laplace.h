#ifndef laplace_h
#define laplace_h
#include <map>
#include <set>
#include "exafmm_t.h"
#include "geometry.h"
#include "intrinsics.h"

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
  void gemm(int m, int n, int k, real_t* A, real_t* B, real_t* C, real_t beta=0.0);

  void svd(int m, int n, real_t* A, real_t* S, real_t* U, real_t* VT);

  RealVec transpose(RealVec& vec, int m, int n);

  void potentialP2P(real_t *src_coord, int src_coord_size, real_t *src_value, real_t *trg_coord, int trg_coord_size, real_t *trg_value);

  void gradientP2P(real_t *src_coord, int src_coord_size, real_t *src_value, real_t *trg_coord, int trg_coord_size, real_t *trg_value);

  void kernelMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out);

  void L2L(Nodes &nodes, RealVec &dnward_equiv, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx);

  void L2P(Nodes& nodes, RealVec &dnward_equiv, std::vector<int> &leafs_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_coord);
  
  void P2P(Nodes &nodes, std::vector<int> leafs_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_pt_src, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_pt_src_idx, int ncrit);

  void M2L(Nodes& nodes, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, RealVec &upward_equiv, RealVec &dnward_equiv);

}//end namespace
#endif
