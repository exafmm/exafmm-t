#include "profile.h"
#include "laplace_cuda.h"
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace exafmm_t {
  
  __global__
  void P2M_kernel(int *d_leafs_idx, int *d_nodes_pt_src_idx, real_t *d_bodies_coord, real_t *d_upward_equiv, int *d_nodes_depth, real_t *d_upwd_check_surf, real_t *d_nodes_coord, real_t *d_nodes_pt_src) {
    int t=threadIdx.x;
    int NSURF = blockDim.x;
    int leaf_idx = d_leafs_idx[blockIdx.x];
    extern __shared__ real_t checkCoord[];
    int level = d_nodes_depth[leaf_idx];
    real_t scal = pow(0.5, level);
    checkCoord[3*t+0] = d_upwd_check_surf[level*NSURF*3+3*t+0] + d_nodes_coord[leaf_idx*3];
    checkCoord[3*t+1] = d_upwd_check_surf[level*NSURF*3+3*t+1] + d_nodes_coord[leaf_idx*3+1] ;
    checkCoord[3*t+2] = d_upwd_check_surf[level*NSURF*3+3*t+2] + d_nodes_coord[leaf_idx*3+2];
    __syncthreads();
    const real_t COEF = 1.0/(2*4*M_PI);
    int src_cnt = (d_nodes_pt_src_idx[blockIdx.x+1]-d_nodes_pt_src_idx[blockIdx.x]);
    real_t *src_coord = &d_bodies_coord[d_nodes_pt_src_idx[blockIdx.x]*3];
    real_t *trg_value = &d_upward_equiv[leaf_idx*NSURF];
    real_t *src_value = &d_nodes_pt_src[d_nodes_pt_src_idx[blockIdx.x]];
    real_t tx = checkCoord[3*t+0];
    real_t ty = checkCoord[3*t+1];
    real_t tz = checkCoord[3*t+2];
    real_t tv = 0;    
    for(int s=0; s<src_cnt; s++) {
      real_t sx = src_coord[3*s+0]-tx;
      real_t sy = src_coord[3*s+1]-ty;
      real_t sz = src_coord[3*s+2]-tz;
      real_t sv = src_value[s];
      real_t r2 = sx*sx + sy*sy + sz*sz;;
      if (r2 != 0) {
        real_t invR = 1.0/sqrt(r2);
        tv += invR * sv;
      }
    }
    tv *= COEF;
    trg_value[t] += tv*scal;
  }

  __global__
  void M2M_kernel(real_t *d_upward_equiv, real_t *d_buffer, int *d_parent_by_level_idx) {
    atomicAdd(&d_upward_equiv[d_parent_by_level_idx[blockIdx.x]*blockDim.x+threadIdx.x], d_buffer[blockIdx.x*blockDim.x+threadIdx.x]);
  }
  
  __global__
  void L2P_kernel(int NSURF, real_t *d_dnward_equiv, real_t *d_bodies_coord, real_t *d_nodes_trg, int *d_leafs_idx, int *d_nodes_pt_src_idx, real_t *d_nodes_coord, int *d_nodes_depth, real_t *d_dnwd_equiv_surf) {
    int i = blockIdx.x;
    int k = threadIdx.x;
    int leaf_idx = d_leafs_idx[i];
    int level = d_nodes_depth[leaf_idx];
    real_t scal = pow(0.5, level);

    extern __shared__ real_t equivCoord_dnward_equiv[];
    real_t *equivCoord = &equivCoord_dnward_equiv[0];
    equivCoord[3*k+0] = d_dnwd_equiv_surf[level*blockDim.x*3+3*k+0] + d_nodes_coord[3*leaf_idx + 0];
    equivCoord[3*k+1] = d_dnwd_equiv_surf[level*blockDim.x*3+3*k+1] + d_nodes_coord[3*leaf_idx + 1];
    equivCoord[3*k+2] = d_dnwd_equiv_surf[level*blockDim.x*3+3*k+2] + d_nodes_coord[3*leaf_idx + 2];
    
    real_t *dnward_equiv = &equivCoord_dnward_equiv[3*blockDim.x];
    dnward_equiv[k] = d_dnward_equiv[leaf_idx*NSURF+k]*scal; 
    __syncthreads();

    int node_start = d_nodes_pt_src_idx[i];
    int node_end = d_nodes_pt_src_idx[i+1];
    real_t *trg_coord = &d_bodies_coord[node_start*3];
    real_t *trg_value = &d_nodes_trg[node_start*4];
    int trg_cnt = node_end-node_start;
    for(int j=0; j< trg_cnt; j++) {  
      const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
      const real_t COEFG = -1.0/(4*2*2*6*M_PI);
      real_t tx = trg_coord[3*j+0];
      real_t ty = trg_coord[3*j+1];
      real_t tz = trg_coord[3*j+2];
      real_t tv0=0;
      real_t tv1=0;
      real_t tv2=0;
      real_t tv3=0;
      real_t sx = equivCoord[3*k+0] - tx;
      real_t sy = equivCoord[3*k+1] - ty;
      real_t sz = equivCoord[3*k+2] - tz;
      real_t r2 = sx*sx + sy*sy + sz*sz;
      real_t sv = dnward_equiv[k];
      if (r2 != 0) {
        real_t invR = 1.0/sqrt(r2);
	real_t invR3 = invR*invR*invR;
	tv0 += invR*sv;
	sv *= invR3;
	tv1 += sv*sx;
        tv2 += sv*sy;
        tv3 += sv*sz;
      }
      tv0 *= COEFP;
      tv1 *= COEFG;
      tv2 *= COEFG;
      tv3 *= COEFG;
      atomicAdd(&trg_value[4*j+0], tv0);
      atomicAdd(&trg_value[4*j+1], tv1);
      atomicAdd(&trg_value[4*j+2], tv2);
      atomicAdd(&trg_value[4*j+3], tv3);
    }
  }

  __global__
  void P2P_kernel(int *d_leafs_idx, int *d_nodes_pt_src_idx, int *d_P2Plists, int *d_P2Plists_idx, real_t *d_bodies_coord, real_t *d_nodes_pt_src, real_t *d_nodes_trg) {
    const real_t COEFP = 1.0/(2*4*M_PI);
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);

    int first_trg_coord_idx = 3*d_nodes_pt_src_idx[blockIdx.x];
    int trg_coord_size = 3*(d_nodes_pt_src_idx[blockIdx.x+1] - d_nodes_pt_src_idx[blockIdx.x]);
    int first_trg_val_idx = 4*first_trg_coord_idx/3;
    if (threadIdx.x < trg_coord_size/3) {
      real_t tx = d_bodies_coord[first_trg_coord_idx+3*threadIdx.x+0];
      real_t ty = d_bodies_coord[first_trg_coord_idx+3*threadIdx.x+1];
      real_t tz = d_bodies_coord[first_trg_coord_idx+3*threadIdx.x+2];
      real_t tv0=0;
      real_t tv1=0;
      real_t tv2=0;
      real_t tv3=0;

      int first_p2plist_idx = d_P2Plists_idx[blockIdx.x];
      int P2Plist_size = d_P2Plists_idx[blockIdx.x+1] - d_P2Plists_idx[blockIdx.x];
      for(int j=0; j<P2Plist_size; j++) {
        int src_idx = d_P2Plists[first_p2plist_idx+j];
        int first_src_coord_idx = 3*d_nodes_pt_src_idx[src_idx];
        int src_coord_size = (d_nodes_pt_src_idx[src_idx+1] - d_nodes_pt_src_idx[src_idx]);
        int first_src_val_idx = d_nodes_pt_src_idx[src_idx];
        for(int k=0; k<src_coord_size; k ++) {
          real_t sx = d_bodies_coord[first_src_coord_idx + 3*k + 0] - tx;
          real_t sy = d_bodies_coord[first_src_coord_idx + 3*k + 1] - ty;
          real_t sz = d_bodies_coord[first_src_coord_idx + 3*k + 2] - tz;
          real_t r2 = sx*sx + sy*sy + sz*sz;
          real_t sv = d_nodes_pt_src[first_src_val_idx+k];
          if (r2 != 0) {
            real_t invR = rsqrtf(r2);
            real_t invR3 = invR*invR*invR;
            tv0 += invR*sv;
            sv *= invR3;
            tv1 += sv*sx;
            tv2 += sv*sy;
            tv3 += sv*sz;
          }
        }
      }
      tv0 *= COEFP;
      tv1 *= COEFG;
      tv2 *= COEFG;
      tv3 *= COEFG;
      d_nodes_trg[first_trg_val_idx+4*threadIdx.x+0] += tv0;
      d_nodes_trg[first_trg_val_idx+4*threadIdx.x+1] += tv1;
      d_nodes_trg[first_trg_val_idx+4*threadIdx.x+2] += tv2;
      d_nodes_trg[first_trg_val_idx+4*threadIdx.x+3] += tv3;
    }
  }
  
  __global__
  void FFT_UpEquiv_kernel(int *d_M2Lsources_idx, int *d_map, real_t *d_up_equiv, real_t *d_upward_equiv, int n3) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int conv_id = d_map[j];
    d_up_equiv[i*n3+conv_id] = d_upward_equiv[d_M2Lsources_idx[i]*blockDim.x+j];
  }

  __global__
  void hadmard_kernel(int *d_M2Ltargets_idx, cufftComplex *d_up_equiv_fft, cufftComplex *d_dw_equiv_fft, int *d_M2LRelPos_offset, int *d_index_in_up_equiv_fft, int *d_M2LRelPoss, real_t *d_mat_M2L_Helper, int n3_, int BLOCKS, int *d_M2Llist_idx_offset, int *d_M2Llist_idx) {
    int i = blockIdx.x;
    int k = threadIdx.x;
    cufftComplex check;
    check.x = 0;
    check.y = 0;

    int d_M2Llist_idx_offset_start = d_M2Llist_idx_offset[i];
    int M2Llist_size = d_M2Llist_idx_offset[i+1]-d_M2Llist_idx_offset_start;
    int M2LRelPos_offset = d_M2LRelPos_offset[i];
    for(int j=0; j <M2Llist_size; j++) {
      int relPosidx = d_M2LRelPoss[M2LRelPos_offset+j];
      real_t *kernel = &d_mat_M2L_Helper[relPosidx*2*n3_];
      cufftComplex *equiv = &d_up_equiv_fft[d_index_in_up_equiv_fft[d_M2Llist_idx[d_M2Llist_idx_offset_start+j]]*n3_];
      int real = 2*k+0;
      int imag = 2*k+1;
      check.x += kernel[real]*equiv[k].x - kernel[imag]*equiv[k].y;
      check.y += kernel[real]*equiv[k].y + kernel[imag]*equiv[k].x;
    }
      d_dw_equiv_fft[i*n3_+k].x += check.x;
      d_dw_equiv_fft[i*n3_+k].y += check.y;
  }

  __global__
  void FFT_Check2Equiv_kernel(int *d_M2Ltargets_idx, int *d_nodes_depth, real_t *d_dnward_equiv, real_t *d_dnCheck, int *d_map, int n3) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int NSURF = blockDim.x;

    int M2Ltarget_idx = d_M2Ltargets_idx[i];
    real_t scale = powf(2, d_nodes_depth[M2Ltarget_idx]);
    int conv_id = d_map[j];
    d_dnward_equiv[M2Ltarget_idx*NSURF+j] += d_dnCheck[i*n3+conv_id] * scale;
  }

  __global__
  void P2L_kernel(real_t *d_dnwd_check_surf, int *d_nodes_P2Llist_idx, int *d_nodes_P2Llist_idx_offset, real_t *d_nodes_coord, int *d_nodes_depth, int *d_nodes_pt_src_idx, real_t *d_bodies_coord, real_t *d_nodes_pt_src, real_t *d_dnward_equiv, int *d_nodes_idx, int NSURF) {
    extern __shared__ real_t targetCheckCoord[];  
    int target_idx = blockIdx.x;
    int source_idx = threadIdx.x;
    int level = d_nodes_depth[d_nodes_idx[target_idx]];
    for(int k=0; k<NSURF; k++) {
      targetCheckCoord[3*k+0] = d_dnwd_check_surf[level*NSURF*3+3*k+0] + d_nodes_coord[d_nodes_idx[target_idx]+0];
      targetCheckCoord[3*k+1] = d_dnwd_check_surf[level*NSURF*3+3*k+1] + d_nodes_coord[d_nodes_idx[target_idx]+1];
      targetCheckCoord[3*k+2] = d_dnwd_check_surf[level*NSURF*3+3*k+2] + d_nodes_coord[d_nodes_idx[target_idx]+2];
    }    
    __syncthreads();
    int src_idx_start = d_nodes_P2Llist_idx_offset[target_idx];
    int src_idx = d_nodes_P2Llist_idx[src_idx_start+source_idx];
    int node_start = d_nodes_pt_src_idx[src_idx];
    int node_end = d_nodes_pt_src_idx[src_idx+1];

    const real_t COEF = 1.0/(2*4*M_PI);
    int src_cnt = node_end-node_start;
    int trg_cnt = NSURF;
    for(int i=0;i<trg_cnt;i++) {
      real_t tx = targetCheckCoord[3*i+0];
      real_t ty = targetCheckCoord[3*i+1];
      real_t tz = targetCheckCoord[3*i+2];
      real_t tv = 0;
      for(int j=0;j<src_cnt;j++) {
        real_t sx = d_bodies_coord[3*j+0]-tx;
        real_t sy = d_bodies_coord[3*j+1]-ty;
        real_t sz = d_bodies_coord[3*j+2]-tz;
        real_t sv = d_nodes_pt_src[node_start+j];
        real_t r2 = sx*sx + sy*sy + sz*sz;;
        if (r2 != 0) {
          real_t invR = 1.0/sqrt(r2);
          tv += invR * sv;
        }
      }
      tv *= COEF;
      d_dnward_equiv[d_nodes_idx[target_idx]*NSURF+i] += tv;
    }
  }
  
  __global__
  void M2P_kernel(int *d_leafs_idx, int *d_nodes_pt_src_idx, int *d_leafs_M2Plist_idx_offset, int *d_leafs_M2Plist_idx, int *d_nodes_depth, real_t *d_nodes_coord, real_t *d_upwd_equiv_surf, real_t *d_upward_equiv, real_t *d_bodies_coord, real_t *d_nodes_trg, int NSURF) {
    int leaf_idx = d_leafs_idx[blockIdx.x];
    int node_start = d_nodes_pt_src_idx[leaf_idx];
    int node_end = d_nodes_pt_src_idx[leaf_idx+1];
    int sources_idx_size = d_leafs_M2Plist_idx_offset[blockIdx.x+1]-d_leafs_M2Plist_idx_offset[blockIdx.x];
    int src_idx_start = d_leafs_M2Plist_idx_offset[blockIdx.x];
    if(threadIdx.x < sources_idx_size) {
      int source_idx = d_leafs_M2Plist_idx[src_idx_start + threadIdx.x];
      int level = d_nodes_depth[source_idx];
      real_t *sourceEquivCoord = new real_t[NSURF*3];
      for(int k=0; k<NSURF; k++) {
        sourceEquivCoord[3*k+0] = d_upwd_equiv_surf[level*NSURF*3+3*k+0] + d_nodes_coord[source_idx + 0];
        sourceEquivCoord[3*k+1] = d_upwd_equiv_surf[level*NSURF*3+3*k+1] + d_nodes_coord[source_idx +1];
        sourceEquivCoord[3*k+2] = d_upwd_equiv_surf[level*NSURF*3+3*k+2] + d_nodes_coord[source_idx +2];
      }
      real_t *trg_coord = &d_bodies_coord[node_start*3];
      real_t *src_coord = &sourceEquivCoord[0];
      real_t *src_value = &d_upward_equiv[source_idx*NSURF];
      real_t *trg_value = &d_nodes_trg[node_start*4];
      const real_t COEFP = 1.0/(2*4*M_PI);   // factor 16 comes from the simd rsqrt function
      const real_t COEFG = -1.0/(4*2*2*6*M_PI);
      int src_cnt = NSURF;
      int trg_cnt = (node_end-node_end);
      for(int i=0; i<trg_cnt; i++) {
        real_t tx = trg_coord[3*i+0];
        real_t ty = trg_coord[3*i+1];
        real_t tz = trg_coord[3*i+2];
        real_t tv0=0;
        real_t tv1=0;
        real_t tv2=0;
        real_t tv3=0;
        for(int j=0; j<src_cnt; j++) {
          real_t sx = src_coord[3*j+0] - tx;
          real_t sy = src_coord[3*j+1] - ty;
          real_t sz = src_coord[3*j+2] - tz;
          real_t r2 = sx*sx + sy*sy + sz*sz;
          real_t sv = src_value[j];
          if (r2 != 0) {
	    real_t invR = 1.0/sqrt(r2);
	    real_t invR3 = invR*invR*invR;
	    tv0 += invR*sv;
	    sv *= invR3;
	    tv1 += sv*sx;
            tv2 += sv*sy;
            tv3 += sv*sz;
          }
        }
        tv0 *= COEFP;
        tv1 *= COEFG;
        tv2 *= COEFG;
        tv3 *= COEFG;
        trg_value[4*i+0] += tv0;
        trg_value[4*i+1] += tv1;
        trg_value[4*i+2] += tv2;
        trg_value[4*i+3] += tv3;
      }
      free(sourceEquivCoord);
    }
  }

  void cuda_init_drivers() {
    cufftHandle init_plan;
    cufftPlan1d(&init_plan, 1, CUFFT_R2C,1);
    cudaFree(0);
  }

  void P2MGPU(real_t *d_upwd_check_surf, int *d_leafs_idx, int *d_nodes_depth, real_t *d_nodes_coord, real_t *d_bodies_coord, int *d_nodes_pt_src_idx, real_t *d_upward_equiv, real_t* d_nodes_pt_src, real_t *d_M2M_V, real_t *d_M2M_U, std::vector<int> &leafs_idx, cublasHandle_t &handle, int BLOCKS, int THREADS) {
    real_t *d_buffer;
    cudaMalloc(&d_buffer, sizeof(real_t)*BLOCKS*NSURF);

    P2M_kernel<<<BLOCKS, THREADS, sizeof(real_t)*NSURF*3>>>(d_leafs_idx, d_nodes_pt_src_idx, d_bodies_coord, d_upward_equiv, d_nodes_depth, d_upwd_check_surf, d_nodes_coord, d_nodes_pt_src);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    real_t alpha=1.0, beta=0.0;
    real_t **M2M_V_p = 0, **upward_equiv_p = 0, **buffer_p = 0, **M2M_U_p;
    M2M_V_p = (real_t**)malloc(BLOCKS * sizeof(real_t*));
    upward_equiv_p = (real_t**)malloc(BLOCKS * sizeof(real_t*));
    buffer_p = (real_t**)malloc(BLOCKS * sizeof(real_t*));
    M2M_U_p = (real_t**)malloc(BLOCKS * sizeof(real_t*));
    #pragma omp parallel for
    for(int i = 0; i < BLOCKS; i++){
      M2M_V_p[i] = d_M2M_V;
      upward_equiv_p[i] = d_upward_equiv + leafs_idx[i]*NSURF;
      buffer_p[i] = d_buffer + i*NSURF;
      M2M_U_p[i] = d_M2M_U;
    }
    real_t **d_M2M_V_p = 0, **d_upward_equiv_p = 0, **d_buffer_p = 0, **d_M2M_U_p=0;
    cudaMalloc(&d_M2M_V_p, BLOCKS*sizeof(real_t*));
    cudaMalloc(&d_upward_equiv_p, BLOCKS*sizeof(real_t*));
    cudaMalloc(&d_buffer_p, BLOCKS*sizeof(real_t*));
    cudaMalloc(&d_M2M_U_p, BLOCKS*sizeof(real_t*));
    cudaMemcpy(d_M2M_V_p, M2M_V_p, sizeof(real_t*)*BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_upward_equiv_p, upward_equiv_p, sizeof(real_t*)*BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer_p, buffer_p, sizeof(real_t*)*BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_U_p, M2M_U_p, sizeof(real_t*)*BLOCKS, cudaMemcpyHostToDevice);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_M2M_V_p, NSURF, (const float**)d_upward_equiv_p, NSURF, &beta, d_buffer_p, NSURF, BLOCKS);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_M2M_U_p, NSURF, (const float**)d_buffer_p, NSURF, &beta, d_upward_equiv_p, NSURF, BLOCKS);

    cudaFree(d_M2M_V_p);
    cudaFree(d_buffer_p);
    cudaFree(d_M2M_U_p);
    free(M2M_V_p);
    free(upward_equiv_p);
    free(buffer_p);
    free(M2M_U_p);
    cudaFree(d_buffer);
  }

  void M2MGPU(real_t *d_upward_equiv, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx, cublasHandle_t &handle) {
    cublasCreate(&handle);
    real_t *d_mat_M2M;
    cudaMalloc(&d_mat_M2M, sizeof(real_t)*mat_M2M.size());    
    
    cudaMemcpy(d_mat_M2M, &mat_M2M[0], sizeof(real_t)*mat_M2M.size(), cudaMemcpyHostToDevice);
    for(int i=nodes_by_level_idx.size()-1;i>=0;i--) {
      real_t *d_buffer;
      
      float **dnward_equiv_p = 0, **mat_M2M_p = 0, **result_p = 0;
      dnward_equiv_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      mat_M2M_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      result_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_buffer, sizeof(real_t)*NSURF*nodes_by_level_idx[i].size());
      for(int j = 0; j < nodes_by_level_idx[i].size(); j++){
          dnward_equiv_p[j] = d_upward_equiv + nodes_by_level_idx[i][j]*NSURF;
          mat_M2M_p[j] = d_mat_M2M + octant_by_level_idx[i][j]*NSURF*NSURF;
          result_p[j] = d_buffer + j*NSURF;
      }
      real_t **d_dnward_equiv_p, **d_mat_M2M_p, **d_result_p;
      cudaMalloc(&d_dnward_equiv_p, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_mat_M2M_p, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_result_p, nodes_by_level_idx[i].size() * sizeof(real_t*));

      cudaMemcpy(d_result_p, result_p, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_mat_M2M_p, mat_M2M_p, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dnward_equiv_p, dnward_equiv_p, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      real_t alpha=1.0, beta=0.0;
      cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_mat_M2M_p, NSURF, (const float**)d_dnward_equiv_p, NSURF, &beta, d_result_p, NSURF, nodes_by_level_idx[i].size());
      int *d_parent_by_level_idx;
      cudaMalloc(&d_parent_by_level_idx, parent_by_level_idx[i].size() * sizeof(int));
      cudaMemcpy(d_parent_by_level_idx, &parent_by_level_idx[i][0], sizeof(int)*parent_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      M2M_kernel<<<parent_by_level_idx[i].size(), NSURF>>>(d_upward_equiv, d_buffer, d_parent_by_level_idx);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      cudaFree(d_buffer);
      cudaFree(d_parent_by_level_idx);
      cudaFree(d_dnward_equiv_p); 
      cudaFree(d_mat_M2M_p);
      free(mat_M2M_p);

    }
    cudaFree(d_mat_M2M);
  }

  void P2PGPU(Nodes &nodes, std::vector<int> &leafs_idx, int *d_leafs_idx, real_t *d_bodies_coord, real_t *d_nodes_pt_src, int *d_nodes_pt_src_idx, real_t *d_nodes_trg, int ncrit) {
    std::vector<int>P2Plist_idx;
    std::vector<int>P2Plist_offset;
    int P2Plists_idx_cnt = 0;

    std::vector<int> targets_idx = leafs_idx;
    for(int i=0; i<targets_idx.size(); i++) {
      Node* target = &nodes[targets_idx[i]];
      std::vector<int> sources_idx = target->P2Plist_idx;
      P2Plist_idx.insert(P2Plist_idx.end(), sources_idx.begin(), sources_idx.end());
      P2Plist_offset.push_back(P2Plists_idx_cnt);
      P2Plists_idx_cnt += sources_idx.size();
    }
    P2Plist_offset.push_back(P2Plists_idx_cnt);

    int BLOCKS = leafs_idx.size();
    int THREADS = ncrit;

    int *d_P2Plists, *d_P2Plists_idx;
    
    cudaMalloc(&d_P2Plists, sizeof(int)*P2Plist_idx.size());
    cudaMalloc(&d_P2Plists_idx, sizeof(int)*P2Plist_offset.size());
    
    cudaMemcpy(d_P2Plists, &P2Plist_idx[0], sizeof(int)*P2Plist_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P2Plists_idx, &P2Plist_offset[0], sizeof(int)*P2Plist_offset.size(), cudaMemcpyHostToDevice);
    P2P_kernel<<<BLOCKS, THREADS>>>(d_leafs_idx, d_nodes_pt_src_idx, d_P2Plists, d_P2Plists_idx, d_bodies_coord, d_nodes_pt_src, d_nodes_trg);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaFree(d_P2Plists_idx);
    cudaFree(d_P2Plists);
  }
  
  cufftComplex *FFT_UpEquiv_GPU(std::vector<int> &M2Lsources_idx, real_t *d_upward_equiv, int upward_equiv_size) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    int dims[] = {n1,n1,n1};
    std::vector<int> map(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf(NSURF*3);
    surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0,0,surf);
    for(size_t i=0; i<map.size(); i++) {
      // mapping: upward equiv surf -> conv grid
      map[i] = ((size_t)(MULTIPOLE_ORDER-1-surf[i*3]+0.5))
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+1]+0.5)) * n1
             + ((size_t)(MULTIPOLE_ORDER-1-surf[i*3+2]+0.5)) * n1 * n1;
    }
    real_t *d_up_equiv;
    int *d_map;
    int* d_M2Lsources_idx;
    cudaMalloc(&d_up_equiv, sizeof(real_t)*M2Lsources_idx.size()*n3);
    cudaMalloc(&d_map, sizeof(int)*map.size());
    cudaMalloc(&d_M2Lsources_idx, sizeof(int)*M2Lsources_idx.size());

    cudaMemcpy(d_map, &map[0], sizeof(int)*map.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2Lsources_idx, &M2Lsources_idx[0], sizeof(int)*M2Lsources_idx.size(), cudaMemcpyHostToDevice);

    FFT_UpEquiv_kernel<<<M2Lsources_idx.size(), NSURF>>>(d_M2Lsources_idx, d_map, d_up_equiv, d_upward_equiv, n3);

    cufftHandle plan_up_equiv;
    cufftPlanMany(&plan_up_equiv, 3, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, M2Lsources_idx.size());
    cufftComplex *d_up_equiv_fft;
    cudaMalloc(&d_up_equiv_fft, sizeof(cufftComplex)*M2Lsources_idx.size()*n3_);
    cufftExecR2C(plan_up_equiv, &d_up_equiv[0], &d_up_equiv_fft[0]);
    cufftDestroy(plan_up_equiv);
    cudaFree(d_up_equiv);
    cudaFree(d_map);
    return &d_up_equiv_fft[0];
  }

  void FFT_Check2Equiv_GPU(Nodes &nodes, cufftComplex *d_dw_equiv_fft, real_t *d_dnward_equiv, int dnward_equiv_size, int *d_M2Ltargets_idx, int M2Ltargets_idx_size, int *d_nodes_depth) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int dims[] = {n1,n1,n1};
    
    real_t *d_dnCheck;
    cudaMalloc(&d_dnCheck, sizeof(real_t)*M2Ltargets_idx_size*n3);
    cufftHandle plan_check_equiv;
    cufftPlanMany(&plan_check_equiv, 3, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, M2Ltargets_idx_size);
    cufftExecC2R(plan_check_equiv, &d_dw_equiv_fft[0], &d_dnCheck[0]);
    cufftDestroy(plan_check_equiv);
    cudaFree(d_dw_equiv_fft);
    std::vector<int> map2(NSURF);
    real_t c[3]= {0, 0, 0};
    for(int d=0; d<3; d++) c[d] += 0.5*(MULTIPOLE_ORDER-2);
    RealVec surf(NSURF*3);
    surface(MULTIPOLE_ORDER, c, (real_t)(MULTIPOLE_ORDER-1), 0,0,surf);
    for(int i=0; i<map2.size(); i++) {
      // mapping: conv grid -> downward check surf
      map2[i] = ((int)(MULTIPOLE_ORDER*2-0.5-surf[i*3]))
              + ((int)(MULTIPOLE_ORDER*2-0.5-surf[i*3+1])) * n1
              + ((int)(MULTIPOLE_ORDER*2-0.5-surf[i*3+2])) * n1 * n1;
    }
    int *d_map;
    cudaMalloc(&d_map, sizeof(int)*map2.size());
    cudaMemcpy(d_map, &map2[0], sizeof(int)*map2.size(), cudaMemcpyHostToDevice);
    FFT_Check2Equiv_kernel<<<M2Ltargets_idx_size, NSURF>>>(d_M2Ltargets_idx, d_nodes_depth, d_dnward_equiv, d_dnCheck, d_map, n3);
    cudaFree(d_dnCheck);
    cudaFree(d_map);
  }

  cufftComplex *HadmardGPU(int *d_M2Ltargets_idx, int M2Ltargets_idx_size, std::vector<int> &M2LRelPos_offset, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, cufftComplex *d_up_equiv_fft, std::vector<int> &M2Llist_idx_offset, std::vector<int> &M2Llist_idx) {
    int BLOCKS = M2Ltargets_idx_size;
    int THREADS = n3_;
    int *d_M2LRelPos_offset, *d_index_in_up_equiv_fft, *d_M2LRelPoss, *d_M2Llist_idx_offset, *d_M2Llist_idx;
    real_t *d_mat_M2L_Helper;
    cufftComplex *d_dw_equiv_fft;
    cudaMalloc(&d_M2LRelPos_offset, sizeof(int)*M2LRelPos_offset.size());
    cudaMalloc(&d_index_in_up_equiv_fft, sizeof(int)*index_in_up_equiv_fft.size());
    cudaMalloc(&d_M2LRelPoss, sizeof(int)*M2LRelPoss.size());
    cudaMalloc(&d_dw_equiv_fft, sizeof(cufftComplex)*M2Ltargets_idx_size*n3_);
    cudaMalloc(&d_mat_M2L_Helper, sizeof(real_t)*mat_M2L_Helper.size());
    cudaMalloc(&d_M2Llist_idx_offset, sizeof(int)*M2Llist_idx_offset.size());
    cudaMalloc(&d_M2Llist_idx, sizeof(int)*M2Llist_idx.size());

    cudaMemcpy(d_M2LRelPos_offset, &M2LRelPos_offset[0], sizeof(int)*M2LRelPos_offset.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_in_up_equiv_fft, &index_in_up_equiv_fft[0], sizeof(int)*index_in_up_equiv_fft.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2LRelPoss, &M2LRelPoss[0], sizeof(int)*M2LRelPoss.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_M2L_Helper, &mat_M2L_Helper[0], sizeof(real_t)*mat_M2L_Helper.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2Llist_idx_offset, &M2Llist_idx_offset[0], sizeof(int)*M2Llist_idx_offset.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2Llist_idx, &M2Llist_idx[0], sizeof(int)*M2Llist_idx.size(), cudaMemcpyHostToDevice);
    Profile::Tic("general",true);
    hadmard_kernel<<<BLOCKS, THREADS>>>(d_M2Ltargets_idx, d_up_equiv_fft, d_dw_equiv_fft, d_M2LRelPos_offset, d_index_in_up_equiv_fft, d_M2LRelPoss, d_mat_M2L_Helper, n3_, BLOCKS, d_M2Llist_idx_offset, d_M2Llist_idx);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Profile::Toc();
    cudaFree(d_M2Llist_idx_offset);
    cudaFree(d_M2Llist_idx);
    cudaFree(d_M2LRelPos_offset);
    cudaFree(d_index_in_up_equiv_fft);
    cudaFree(d_M2LRelPoss);
    cudaFree(d_up_equiv_fft);
    cudaFree(d_mat_M2L_Helper);
    return &d_dw_equiv_fft[0];
  }

  void M2LGPU(Nodes &nodes, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, real_t *d_upward_equiv, int upward_equiv_size, real_t *d_dnward_equiv, int dnward_equiv_size, int *d_nodes_depth) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    std::vector<int> index_in_up_equiv_fft(nodes.size());
    std::vector<int> M2LRelPos_offset;
    std::vector<int> M2LRelPoss;
    int M2LRelPos_offset_cnt = 0;
    #pragma omp parallel for
    for(int i=0; i<M2Lsources_idx.size(); ++i)
      index_in_up_equiv_fft[M2Lsources_idx[i]] = i;
    
    std::vector<int> M2Llist_idx;
    std::vector<int> M2Llist_idx_offset;
    int M2Llist_idx_cnt = 0;
    for (int i=0;i<M2Ltargets_idx.size(); i++) {
      Node* target = &nodes[M2Ltargets_idx[i]];
      M2LRelPos_offset.push_back(M2LRelPos_offset_cnt);
      M2Llist_idx.insert(M2Llist_idx.end(), target->M2Llist_idx.begin(), target->M2Llist_idx.end());
      M2Llist_idx_offset.push_back(M2Llist_idx_cnt);
      M2Llist_idx_cnt += target->M2Llist_idx.size();
      for(int j=0; j<target->M2Llist_idx.size(); j++) {
        Node* source = &nodes[target->M2Llist_idx[j]];
        M2LRelPoss.push_back(target->M2LRelPos[j]);
        M2LRelPos_offset_cnt ++;
      }
    }
    M2Llist_idx_offset.push_back(M2Llist_idx_cnt);
    M2LRelPos_offset.push_back(M2LRelPos_offset_cnt);
    int *d_M2Ltargets_idx;
    cudaMalloc(&d_M2Ltargets_idx, sizeof(int)*M2Ltargets_idx.size());
    cudaMemcpy(d_M2Ltargets_idx, &M2Ltargets_idx[0], sizeof(int)*M2Ltargets_idx.size(), cudaMemcpyHostToDevice);
    cufftComplex *d_up_equiv_fft = FFT_UpEquiv_GPU(M2Lsources_idx, d_upward_equiv, upward_equiv_size);
    cufftComplex *d_dw_equiv_fft = HadmardGPU(d_M2Ltargets_idx, M2Ltargets_idx.size(), M2LRelPos_offset, index_in_up_equiv_fft, M2LRelPoss, mat_M2L_Helper, n3_, d_up_equiv_fft, M2Llist_idx_offset, M2Llist_idx);
    FFT_Check2Equiv_GPU(nodes, d_dw_equiv_fft, d_dnward_equiv, dnward_equiv_size, d_M2Ltargets_idx, M2Ltargets_idx.size(), d_nodes_depth);
    cudaFree(d_M2Ltargets_idx);
  }
  
  void L2PGPU(Nodes &nodes, real_t *d_dnward_equiv, std::vector<int> &leafs_idx, int *d_leafs_idx, int leafs_idx_size, real_t *d_nodes_trg, int *d_nodes_pt_src_idx, real_t *d_bodies_coord, real_t *d_nodes_coord, int *d_nodes_depth, real_t *d_dnwd_equiv_surf, cublasHandle_t &handle) {
    real_t *d_L2L_U, *d_buffer, *d_L2L_V;

    cudaMalloc(&d_L2L_U, sizeof(real_t)*L2L_U.size());
    cudaMalloc(&d_buffer, sizeof(real_t)*NSURF*leafs_idx_size);
    cudaMalloc(&d_L2L_V, sizeof(real_t)*L2L_V.size());

    cudaMemcpy(d_L2L_U, &L2L_U[0], sizeof(real_t)*L2L_U.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L2L_V, &L2L_V[0], sizeof(real_t)*L2L_V.size(), cudaMemcpyHostToDevice);

    real_t alpha=1.0, beta=0.0;
    real_t **dnward_equiv_p=0, **buffer_p=0, **L2L_V_p=0, **L2L_U_p=0;
    dnward_equiv_p = (real_t**)malloc(leafs_idx_size * sizeof(real_t*));
    buffer_p = (real_t**)malloc(leafs_idx_size * sizeof(real_t*));
    L2L_V_p = (real_t**)malloc(leafs_idx_size * sizeof(real_t*));
    L2L_U_p = (real_t**)malloc(leafs_idx_size * sizeof(real_t*));
    for(int i = 0; i < leafs_idx_size; i++){
      dnward_equiv_p[i] = d_dnward_equiv+leafs_idx[i]*NSURF;
      buffer_p[i] = d_buffer+i*NSURF;
      L2L_V_p[i] = d_L2L_V;
      L2L_U_p[i] =d_L2L_U;
    }
    real_t **d_dnward_equiv_p=0, **d_buffer_p=0, **d_L2L_V_p=0, **d_L2L_U_p=0;
    cudaMalloc(&d_dnward_equiv_p, leafs_idx_size*sizeof(real_t*));
    cudaMalloc(&d_buffer_p, leafs_idx_size*sizeof(real_t*));
    cudaMalloc(&d_L2L_V_p, leafs_idx_size*sizeof(real_t*));
    cudaMalloc(&d_L2L_U_p, leafs_idx_size*sizeof(real_t*));

    cudaMemcpy(d_dnward_equiv_p, dnward_equiv_p, sizeof(real_t*)*leafs_idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer_p, buffer_p, sizeof(real_t*)*leafs_idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L2L_V_p, L2L_V_p, sizeof(real_t*)*leafs_idx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L2L_U_p, L2L_U_p, sizeof(real_t*)*leafs_idx_size, cudaMemcpyHostToDevice);

    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_L2L_V_p, NSURF, (const float**)d_dnward_equiv_p, NSURF, &beta, d_buffer_p, NSURF, leafs_idx.size());
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_L2L_U_p, NSURF, (const float**)d_buffer_p, NSURF, &beta, d_dnward_equiv_p, NSURF, leafs_idx_size);
    int BLOCKS = leafs_idx_size;
    L2P_kernel<<<BLOCKS, NSURF, sizeof(real_t)*NSURF*4>>>(NSURF, d_dnward_equiv, d_bodies_coord, d_nodes_trg, d_leafs_idx, d_nodes_pt_src_idx, d_nodes_coord, d_nodes_depth, d_dnwd_equiv_surf);
    free(dnward_equiv_p);
    free(buffer_p);
    free(L2L_V_p);
    free(L2L_U_p);
    cudaFree(d_dnward_equiv_p);
    cudaFree(d_buffer_p);
    cudaFree(d_L2L_V_p);
    cudaFree(d_L2L_U_p);
    cudaFree(d_L2L_U);
    cudaFree(d_buffer);
    cudaFree(d_L2L_V);
  }

  void L2LGPU(Nodes &nodes, real_t *d_dnward_equiv, int dnward_equiv_size, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx, cublasHandle_t &handle) {
    real_t *d_mat_L2L;
    cudaMalloc(&d_mat_L2L, sizeof(real_t)*mat_L2L.size());

    cudaMemcpy(d_mat_L2L, &mat_L2L[0], sizeof(real_t)*mat_L2L.size(), cudaMemcpyHostToDevice);
    
    for(int i=0; i<nodes_by_level_idx.size(); i++) {
      float **dnward_equiv_p=0, **mat_L2L_p=0, **result_p=0;
      dnward_equiv_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      mat_L2L_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      result_p = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*
));
      for(int j=0; j<nodes_by_level_idx[i].size();j++) {
        int node_idx = nodes_by_level_idx[i][j];
        int parent_idx = parent_by_level_idx[i][j];
        int octant = octant_by_level_idx[i][j];
        dnward_equiv_p[j] = d_dnward_equiv+parent_idx*NSURF;
        mat_L2L_p[j] = d_mat_L2L+octant*NSURF*NSURF;
        result_p[j] = d_dnward_equiv+node_idx*NSURF;
      }
      real_t **d_dnward_equiv_p, **d_mat_L2L_p, **d_result_p;
      cudaMalloc(&d_dnward_equiv_p, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_mat_L2L_p, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_result_p, nodes_by_level_idx[i].size() * sizeof(real_t*));
      
      cudaMemcpy(d_mat_L2L_p, mat_L2L_p, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dnward_equiv_p, dnward_equiv_p,  sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_result_p, result_p, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      
      real_t alpha=1.0, beta=1.0;
      cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_mat_L2L_p, NSURF, (const float**)d_dnward_equiv_p, NSURF, &beta, d_result_p, NSURF, nodes_by_level_idx[i].size());
      free(dnward_equiv_p);
      free(mat_L2L_p);
      free(result_p);
      cudaFree(d_dnward_equiv_p);
      cudaFree(d_mat_L2L_p);
      cudaFree(d_result_p);
    }
    cudaFree(d_mat_L2L);
  }
  
  void P2LGPU(Nodes& nodes, real_t *d_dnward_equiv, real_t *d_nodes_pt_src, int *d_nodes_pt_src_idx, real_t *d_bodies_coord, real_t *d_nodes_coord, int *d_nodes_depth, int *d_nodes_idx, real_t *d_dnwd_check_surf) {
    int sources_max = 0;
    for(int i=0; i<nodes.size(); i++) {
      sources_max = (nodes[i].P2Llist_idx.size() > sources_max)? nodes[i].P2Llist_idx.size():sources_max;
    }
    
    if(sources_max > 0) {
      std::vector<int> nodes_P2Llist_idx;
      std::vector<int> nodes_P2Llist_idx_offset;
      int  nodes_P2Llist_idx_offset_cnt = 0;
      for(int i=0; i<nodes.size(); i++) {
        std::vector<int> sources_idx = nodes[i].P2Llist_idx;
        nodes_P2Llist_idx.insert(nodes_P2Llist_idx.end(), sources_idx.begin(), sources_idx.end());
        nodes_P2Llist_idx_offset.push_back(nodes_P2Llist_idx_offset_cnt);
        nodes_P2Llist_idx_offset_cnt += sources_idx.size();
      }
      nodes_P2Llist_idx_offset.push_back(nodes_P2Llist_idx_offset_cnt);

      int THREADS = sources_max;
      int BLOCKS = nodes.size();
      int *d_nodes_P2Llist_idx, *d_nodes_P2Llist_idx_offset;
    
      cudaMalloc(&d_nodes_P2Llist_idx, sizeof(int)*nodes_P2Llist_idx.size());
      cudaMalloc(&d_nodes_P2Llist_idx_offset, sizeof(int)*nodes_P2Llist_idx_offset.size());

      cudaMemcpy(d_nodes_P2Llist_idx, &nodes_P2Llist_idx[0], sizeof(int)*nodes_P2Llist_idx.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_nodes_P2Llist_idx_offset, &nodes_P2Llist_idx_offset[0], sizeof(int)*nodes_P2Llist_idx_offset.size(), cudaMemcpyHostToDevice);
    
      P2L_kernel<<<BLOCKS, THREADS, NSURF*3*sizeof(real_t)>>>(d_dnwd_check_surf, d_nodes_P2Llist_idx, d_nodes_P2Llist_idx_offset, d_nodes_coord, d_nodes_depth, d_nodes_pt_src_idx, d_bodies_coord, d_nodes_pt_src, d_dnward_equiv, d_nodes_idx, NSURF);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      
      cudaFree(d_nodes_P2Llist_idx);
      cudaFree(d_nodes_P2Llist_idx_offset);
    }
  }
  
  void M2PGPU(Nodes &nodes, std::vector<int> leafs_idx, int *d_leafs_idx, real_t *d_upward_equiv, real_t *d_nodes_trg, int *d_nodes_pt_src_idx, real_t *d_bodies_coord, int *d_nodes_depth, real_t *d_nodes_coord, real_t *d_upwd_equiv_surf, int BLOCKS) {
    int sources_max = 0;
    for(int i=0; i<BLOCKS; i++) {
      Node* target = &nodes[leafs_idx[i]];
      std::vector<int> sources_idx= target->M2Plist_idx;
      sources_max = (sources_idx.size() > sources_max)? sources_idx.size():sources_max;
    }

    if(sources_max>0) {
      std::vector<int> leafs_M2Plist_idx;
      std::vector<int> leafs_M2Plist_idx_offset;
      int leafs_M2Plist_idx_offset_cnt = 0;
      for(int i=0; i<BLOCKS; i++) {
        Node* target = &nodes[leafs_idx[i]];
        std::vector<int> sources_idx= target->M2Plist_idx;
        leafs_M2Plist_idx.insert(leafs_M2Plist_idx.end(), sources_idx.begin(), sources_idx.end());
        leafs_M2Plist_idx_offset.push_back(leafs_M2Plist_idx_offset_cnt);
        leafs_M2Plist_idx_offset_cnt += sources_idx.size();
      }
      leafs_M2Plist_idx_offset.push_back(leafs_M2Plist_idx_offset_cnt);

      int THREADS = sources_max;

      int *d_leafs_M2Plist_idx_offset, *d_leafs_M2Plist_idx;
    
      cudaMalloc(&d_leafs_M2Plist_idx_offset, sizeof(int)*leafs_M2Plist_idx_offset.size());
      cudaMalloc(&d_leafs_M2Plist_idx, sizeof(int)*leafs_M2Plist_idx.size());

      cudaMemcpy(d_leafs_M2Plist_idx_offset, &leafs_M2Plist_idx_offset[0], sizeof(int)*leafs_M2Plist_idx_offset.size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_leafs_M2Plist_idx, &leafs_M2Plist_idx[0], sizeof(int)*leafs_M2Plist_idx.size(), cudaMemcpyHostToDevice);
    
      M2P_kernel<<<BLOCKS, THREADS>>>(d_leafs_idx, d_nodes_pt_src_idx, d_leafs_M2Plist_idx_offset, d_leafs_M2Plist_idx, d_nodes_depth, d_nodes_coord, d_upwd_equiv_surf, d_upward_equiv, d_bodies_coord, d_nodes_trg, NSURF);

      cudaFree(d_leafs_M2Plist_idx_offset);
      cudaFree(d_leafs_M2Plist_idx);
    }
  }
  
  void fmmStepsGPU(Nodes& nodes, std::vector<int> &leafs_idx, std::vector<real_t> &bodies_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, int ncrit, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx, std::vector<real_t> &nodes_coord, std::vector<int> &M2Lsources_idx, std::vector<int> &M2Ltargets_idx, std::vector<real_t> &nodes_trg, std::vector<int> &nodes_depth, std::vector<int> &nodes_idx) {
    Profile::Tic("totalgpu", true);
    real_t c[3] = {0.0};
    std::vector<real_t> upwd_check_surf((MAXLEVEL+1)*NSURF*3);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      surface(MULTIPOLE_ORDER,c,2.95,depth, depth, upwd_check_surf);
    }
    std::vector<real_t> dnwd_check_surf((MAXLEVEL+1)*NSURF*3);
    for(size_t depth = 0; depth <= MAXLEVEL; depth++) {
      surface(MULTIPOLE_ORDER,c,1.05,depth,depth,dnwd_check_surf);
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    int *d_nodes_pt_src_idx, *d_leafs_idx, *d_nodes_depth, *d_nodes_idx;
    real_t *d_bodies_coord, *d_upward_equiv, *d_M2M_V, *d_M2M_U, *d_nodes_coord, *d_nodes_pt_src, *d_dnward_equiv, *d_upwd_check_surf, *d_nodes_trg, *d_dnwd_check_surf;
    Profile::Tic("memcopying",true);
    cudaMalloc(&d_nodes_trg, sizeof(real_t)*nodes_trg.size());
    cudaMalloc(&d_dnwd_check_surf, sizeof(real_t)*dnwd_check_surf.size());
    cudaMalloc(&d_nodes_idx, sizeof(int)*nodes_idx.size());
    cudaMalloc(&d_nodes_depth, sizeof(int)*nodes_depth.size());
    cudaMalloc(&d_nodes_pt_src_idx, sizeof(int)*nodes_pt_src_idx.size());
    cudaMalloc(&d_bodies_coord, sizeof(real_t)*bodies_coord.size());
    cudaMalloc(&d_upward_equiv, sizeof(real_t)*nodes.size()*NSURF);
    cudaMalloc(&d_M2M_V, sizeof(real_t)*M2M_V.size());
    cudaMalloc(&d_M2M_U, sizeof(real_t)*M2M_U.size());
    cudaMalloc(&d_leafs_idx, sizeof(int)*leafs_idx.size());
    cudaMalloc(&d_nodes_coord, sizeof(real_t)*nodes_coord.size());
    cudaMalloc(&d_nodes_pt_src, sizeof(real_t)*nodes_pt_src.size());
    cudaMalloc(&d_upwd_check_surf, sizeof(real_t)*upwd_check_surf.size());
    cudaMalloc(&d_dnward_equiv, sizeof(real_t)*nodes.size()*NSURF);

    cudaMemcpy(d_dnwd_check_surf, &dnwd_check_surf[0], sizeof(int)*dnwd_check_surf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_idx, &nodes_idx[0], sizeof(int)*nodes_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_depth, &nodes_depth[0], sizeof(int)*nodes_depth.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leafs_idx, &leafs_idx[0], sizeof(int)*leafs_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_U, &M2M_U[0], sizeof(real_t)*M2M_U.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_V, &M2M_V[0], sizeof(real_t)*M2M_V.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src_idx, &nodes_pt_src_idx[0], sizeof(int)*nodes_pt_src_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies_coord, &bodies_coord[0], sizeof(real_t)*bodies_coord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_coord, &nodes_coord[0], sizeof(real_t)*nodes_coord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src, &nodes_pt_src[0], sizeof(real_t)*nodes_pt_src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upwd_check_surf, &upwd_check_surf[0], sizeof(real_t)*upwd_check_surf.size(), cudaMemcpyHostToDevice);
    Profile::Toc();
    Profile::Tic("P2M", false, 5);
    P2MGPU(d_upwd_check_surf, d_leafs_idx, d_nodes_depth, d_nodes_coord, d_bodies_coord, d_nodes_pt_src_idx, d_upward_equiv, d_nodes_pt_src, d_M2M_V, d_M2M_U, leafs_idx, handle, leafs_idx.size(), NSURF);
    Profile::Toc();
   
    Profile::Tic("M2M", false, 5);
    M2MGPU(d_upward_equiv, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, handle);
    Profile::Toc();
    
    Profile::Tic("P2L", false, 5);
    P2LGPU(nodes, d_dnward_equiv, d_nodes_pt_src, d_nodes_pt_src_idx, d_bodies_coord, d_nodes_coord, d_nodes_depth, d_nodes_idx, d_dnwd_check_surf);
    Profile::Toc();
    
    Profile::Tic("M2P", false, 5);
    M2PGPU(nodes, leafs_idx, d_leafs_idx, d_upward_equiv, d_nodes_trg, d_nodes_pt_src_idx, d_bodies_coord, d_nodes_depth, d_nodes_coord, d_dnwd_check_surf, leafs_idx.size());
    Profile::Toc();
    
    Profile::Tic("P2P", false, 5);
    P2PGPU(nodes, leafs_idx, d_leafs_idx, d_bodies_coord, d_nodes_pt_src, d_nodes_pt_src_idx, d_nodes_trg, ncrit);
    Profile::Toc();
    
    Profile::Tic("M2L", false, 5);
    M2LGPU(nodes, M2Lsources_idx, M2Ltargets_idx, d_upward_equiv, nodes.size()*NSURF, d_dnward_equiv, nodes.size()*NSURF, d_nodes_depth);
    Profile::Toc();

    Profile::Tic("L2L", false, 5);
    L2LGPU(nodes, d_dnward_equiv, nodes.size()*NSURF, nodes_by_level_idx, parent_by_level_idx, octant_by_level_idx, handle);
    Profile::Toc();
    
    Profile::Tic("L2P", false, 5);
    L2PGPU(nodes, d_dnward_equiv, leafs_idx, d_leafs_idx, leafs_idx.size(), d_nodes_trg, d_nodes_pt_src_idx, d_bodies_coord, d_nodes_coord, d_nodes_depth, d_upwd_check_surf, handle);
    Profile::Toc();
    cudaMemcpy(&nodes_trg[0], d_nodes_trg, sizeof(real_t)*nodes_trg.size(), cudaMemcpyDeviceToHost);
   
    cudaFree(d_leafs_idx);
    cudaFree(d_M2M_U);
    cudaFree(d_M2M_V);
    cudaFree(d_nodes_pt_src_idx);
    cudaFree(d_bodies_coord);
    cudaFree(d_upward_equiv);
    cudaFree(d_nodes_depth);
    cudaFree(d_nodes_coord);
    cudaFree(d_nodes_pt_src);
    cudaFree(d_dnward_equiv);
    cudaFree(d_nodes_idx);
    cudaFree(d_dnwd_check_surf);
    cudaFree(d_nodes_trg);
    cudaFree(d_upwd_check_surf);
    Profile::Toc();
  }
}


