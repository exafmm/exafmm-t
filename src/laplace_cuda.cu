#include "exafmm_t.h"
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
  void P2M_potential_p2p_kernel(int *leafs_idx, int *d_nodes_pt_src_idx, real_t *d_nodes_coord, real_t *d_nodes_pt_src, real_t *d_checkCoord, real_t *d_upward_equiv, real_t *d_r, real_t *d_leaf_xyz) {
    int i = blockIdx.x;
    int t = threadIdx.x;
    const real_t COEF = 1.0/(2*4*M_PI);
    real_t tx = d_checkCoord[3*t+0]*d_r[i]+d_leaf_xyz[3*i+0];
    real_t ty = d_checkCoord[3*t+1]*d_r[i]+d_leaf_xyz[3*i+1];
    real_t tz = d_checkCoord[3*t+2]*d_r[i]+d_leaf_xyz[3*i+2];
    real_t tv = 0;
    int src_cnt = (d_nodes_pt_src_idx[leafs_idx[i]+1]-d_nodes_pt_src_idx[leafs_idx[i]]);
    int leaf_pt_src_idx = d_nodes_pt_src_idx[leafs_idx[i]];
    int leaf_coord_idx = 3*leaf_pt_src_idx;
    for(int s=0; s<src_cnt; s++) {
      real_t sx = d_nodes_coord[leaf_coord_idx+3*s+0]-tx;
      real_t sy = d_nodes_coord[leaf_coord_idx+3*s+1]-ty;
      real_t sz = d_nodes_coord[leaf_coord_idx+3*s+2]-tz;
      real_t sv = d_nodes_pt_src[leaf_pt_src_idx+s];
      real_t r2 = sx*sx + sy*sy + sz*sz;;
      if (r2 != 0) {
        real_t invR = rsqrtf(r2);
        tv += invR * sv;
      }
    }
    tv *= COEF;
    d_upward_equiv[leafs_idx[i]*blockDim.x+t] += tv;
  }

  __global__
  void M2MKernel(real_t *d_upward_equiv, real_t *d_buffer, int *d_parent_by_level_idx) {
    atomicAdd(&d_upward_equiv[d_parent_by_level_idx[blockIdx.x]*blockDim.x+threadIdx.x], d_buffer[blockIdx.x*blockDim.x+threadIdx.x]);
  }

 __global__
  void gradientP2PKernel(int *d_leafs_idx, int *d_nodes_pt_src_idx, int *d_P2Plists, int *d_P2Plists_idx, real_t *d_nodes_coord, real_t *d_nodes_pt_src, real_t *d_trg_val) {
    const real_t COEFP = 1.0/(2*4*M_PI);
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);

    int first_trg_coord_idx = 3*d_nodes_pt_src_idx[d_leafs_idx[blockIdx.x]];
    int trg_coord_size = 3*(d_nodes_pt_src_idx[d_leafs_idx[blockIdx.x]+1] - d_nodes_pt_src_idx[d_leafs_idx[blockIdx.x]]);
    int first_trg_val_idx = 4*first_trg_coord_idx/3;
    if (threadIdx.x < trg_coord_size/3) {
      real_t tx = d_nodes_coord[first_trg_coord_idx+3*threadIdx.x+0];
      real_t ty = d_nodes_coord[first_trg_coord_idx+3*threadIdx.x+1];
      real_t tz = d_nodes_coord[first_trg_coord_idx+3*threadIdx.x+2];
      real_t tv0=0;
      real_t tv1=0;
      real_t tv2=0;
      real_t tv3=0;

      int first_p2plist_idx = d_P2Plists_idx[blockIdx.x];
      int P2Plist_size = d_P2Plists_idx[blockIdx.x+1] - d_P2Plists_idx[blockIdx.x];
      for(int j=0; j<P2Plist_size; j++) {
        int src_idx = d_P2Plists[first_p2plist_idx+j];
        int first_src_coord_idx = 3*d_nodes_pt_src_idx[src_idx];
        int src_coord_size = 3*(d_nodes_pt_src_idx[src_idx+1] - d_nodes_pt_src_idx[src_idx]);
        int first_src_val_idx = d_nodes_pt_src_idx[src_idx];
        for(int k=0; k<src_coord_size/3; k ++) {
          real_t sx = d_nodes_coord[first_src_coord_idx + 3*k + 0] - tx;
          real_t sy = d_nodes_coord[first_src_coord_idx + 3*k + 1] - ty;
          real_t sz = d_nodes_coord[first_src_coord_idx + 3*k + 2] - tz;
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
      d_trg_val[first_trg_val_idx+4*threadIdx.x+0] += tv0;
      d_trg_val[first_trg_val_idx+4*threadIdx.x+1] += tv1;
      d_trg_val[first_trg_val_idx+4*threadIdx.x+2] += tv2;
      d_trg_val[first_trg_val_idx+4*threadIdx.x+3] += tv3;
    }
  }
   
  __global__
  void hadmard_kernel(int *d_M2Ltargets_idx, cufftComplex *d_up_equiv_fft, cufftComplex *d_dw_equiv_fft, int *d_M2LRelPos_start_idx, int *d_index_in_up_equiv_fft, int *d_M2LRelPoss, real_t *d_mat_M2L_Helper, int n3_, int BLOCKS) {
    int i = blockIdx.x;
    int k = threadIdx.x;
    int M2LRelPos_size = d_M2LRelPos_start_idx[i+1]-d_M2LRelPos_start_idx[i];
    for(int j=0; j <M2LRelPos_size; j++) {
      int relPosidx = d_M2LRelPoss[d_M2LRelPos_start_idx[i]+j];
      real_t *kernel = &d_mat_M2L_Helper[relPosidx*2*n3_];
      cufftComplex *equiv = &d_up_equiv_fft[d_index_in_up_equiv_fft[d_M2LRelPos_start_idx[i]+j]*n3_];
      cufftComplex *check = &d_dw_equiv_fft[i*n3_];
      int real = 2*k+0;
      int imag = 2*k+1;
      check[k].x += kernel[real]*equiv[k].x - kernel[imag]*equiv[k].y;
      check[k].y += kernel[real]*equiv[k].y + kernel[imag]*equiv[k].x;
    }
  }

  void cuda_init_drivers() {
    cudaFree(0);
}
  
  void P2MGPU(std::vector<int> &leafs_idx, std::vector<real_t> &nodes_coord, std::vector<real_t> &nodes_pt_src, std::vector<int> &nodes_pt_src_idx, std::vector<real_t> &checkCoord, int trg_cnt, RealVec &upward_equiv, std::vector<real_t> &r, std::vector<real_t> &leaf_xyz, int leafs_size, int ncrit) {
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);

    int BLOCKS = leafs_size;
    int THREADS = trg_cnt/3; 
    int *d_nodes_pt_src_idx, *d_leafs_idx;
    real_t *d_nodes_coord, *d_nodes_pt_src, *d_checkCoord, *d_upward_equiv, *d_r, *d_leaf_xyz, *d_M2M_V, *d_buffer, *d_M2M_U;

    cudaMalloc(&d_nodes_pt_src_idx, sizeof(int)*nodes_pt_src_idx.size());
    cudaMalloc(&d_nodes_coord, sizeof(real_t)*nodes_coord.size());
    cudaMalloc(&d_nodes_pt_src, sizeof(real_t)*nodes_pt_src.size());
    cudaMalloc(&d_checkCoord, sizeof(real_t)*checkCoord.size());
    cudaMalloc(&d_upward_equiv, sizeof(real_t)*upward_equiv.size());
    cudaMalloc(&d_r, sizeof(real_t)*r.size());
    cudaMalloc(&d_leaf_xyz, sizeof(real_t)*leaf_xyz.size());
    cudaMalloc(&d_M2M_V, sizeof(real_t)*M2M_V.size());
    cudaMalloc(&d_buffer, sizeof(real_t)*leafs_idx.size()*NSURF);
    cudaMalloc(&d_M2M_U, sizeof(real_t)*M2M_U.size());
    cudaMalloc(&d_leafs_idx, sizeof(int)*leafs_idx.size());

    cudaMemcpy(d_leafs_idx, &leafs_idx[0], sizeof(int)*leafs_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_U, &M2M_U[0], sizeof(real_t)*M2M_U.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_V, &M2M_V[0], sizeof(real_t)*M2M_V.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src_idx, &nodes_pt_src_idx[0], sizeof(int)*nodes_pt_src_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_coord, &nodes_coord[0], sizeof(real_t)*nodes_coord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src, &nodes_pt_src[0], sizeof(real_t)*nodes_pt_src.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_checkCoord, &checkCoord[0], sizeof(real_t)*checkCoord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upward_equiv, &upward_equiv[0], sizeof(real_t)*upward_equiv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, &r[0], sizeof(real_t)*r.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaf_xyz, &leaf_xyz[0], sizeof(real_t)*leaf_xyz.size(), cudaMemcpyHostToDevice);
    Profile::Tic("general",true);
    P2M_potential_p2p_kernel<<<BLOCKS, THREADS>>>(d_leafs_idx, d_nodes_pt_src_idx, d_nodes_coord, d_nodes_pt_src, d_checkCoord, d_upward_equiv, d_r, d_leaf_xyz);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    real_t alpha=1.0, beta=0.0;
    real_t **M2M_V_p = 0, **upward_equiv_p = 0, **buffer_p = 0, **M2M_U_p;
    M2M_V_p = (real_t**)malloc(leafs_size * sizeof(real_t*));
    upward_equiv_p = (real_t**)malloc(leafs_size * sizeof(real_t*));
    buffer_p = (real_t**)malloc(leafs_size * sizeof(real_t*));
    M2M_U_p = (real_t**)malloc(leafs_size * sizeof(real_t*));
    for(int i = 0; i < leafs_size; i++){
      M2M_V_p[i] = d_M2M_V;
      upward_equiv_p[i] = d_upward_equiv + leafs_idx[i]*NSURF;
      buffer_p[i] = d_buffer + i*NSURF;
      M2M_U_p[i] = d_M2M_U;
    }
    real_t **d_M2M_V_p = 0, **d_upward_equiv_p = 0, **d_buffer_p = 0, **d_M2M_U_p=0;
    cudaMalloc(&d_M2M_V_p, leafs_size*sizeof(real_t*));
    cudaMalloc(&d_upward_equiv_p, leafs_size*sizeof(real_t*));
    cudaMalloc(&d_buffer_p, leafs_size*sizeof(real_t*));
    cudaMalloc(&d_M2M_U_p, leafs_size*sizeof(real_t*));
    cudaMemcpy(d_M2M_V_p, M2M_V_p, sizeof(real_t*)*leafs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_upward_equiv_p, upward_equiv_p, sizeof(real_t*)*leafs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer_p, buffer_p, sizeof(real_t*)*leafs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2M_U_p, M2M_U_p, sizeof(real_t*)*leafs_size, cudaMemcpyHostToDevice);

    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_M2M_V_p, NSURF, (const float**)d_upward_equiv_p, NSURF, &beta, d_buffer_p, NSURF, leafs_size);
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_M2M_U_p, NSURF, (const float**)d_buffer_p, NSURF, &beta, d_upward_equiv_p, NSURF, leafs_size);
    Profile::Toc();
    cudaMemcpy(&upward_equiv[0], d_upward_equiv, sizeof(real_t)*upward_equiv.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_nodes_pt_src_idx);
    cudaFree(d_leafs_idx);
    cudaFree(d_nodes_coord);
    cudaFree(d_nodes_pt_src);
    cudaFree(d_checkCoord);
    cudaFree(d_upward_equiv);
    cudaFree(d_r);
    cudaFree(d_leaf_xyz);
    cudaFree(d_M2M_U);
    cudaFree(d_M2M_V);
    cudaFree(d_buffer);
    cublasDestroy(handle);
  }

void M2MGPU(RealVec &upward_equiv, std::vector<std::vector<int>> &nodes_by_level_idx, std::vector<std::vector<int>> &parent_by_level_idx, std::vector<std::vector<int>> &octant_by_level_idx) {
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    real_t *d_mat_M2M, *d_upward_equiv;
    cudaMalloc(&d_upward_equiv, sizeof(real_t)*upward_equiv.size());
    cudaMalloc(&d_mat_M2M, sizeof(real_t)*mat_M2M.size());    
    
    cudaMemcpy(d_mat_M2M, &mat_M2M[0], sizeof(real_t)*mat_M2M.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upward_equiv, &upward_equiv[0], sizeof(real_t)*upward_equiv.size(), cudaMemcpyHostToDevice);
    for(int i=nodes_by_level_idx.size()-1;i>=0;i--) {
      real_t *d_buffer;
      
      float **AList = 0, **BList = 0, **CList = 0;
      AList = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      BList = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      CList = (real_t**)malloc(nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_buffer, sizeof(real_t)*NSURF*nodes_by_level_idx[i].size());
      for(int j = 0; j < nodes_by_level_idx[i].size(); j++){
          AList[j] = d_upward_equiv + nodes_by_level_idx[i][j]*NSURF;
          BList[j] = d_mat_M2M + octant_by_level_idx[i][j]*NSURF*NSURF;
          CList[j] = d_buffer + j*NSURF;
      }
      real_t **d_AList, **d_BList, **d_CList;
      cudaMalloc(&d_AList, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_BList, nodes_by_level_idx[i].size() * sizeof(real_t*));
      cudaMalloc(&d_CList, nodes_by_level_idx[i].size() * sizeof(real_t*));

      cudaMemcpy(d_CList, CList, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_BList, BList, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      cudaMemcpy(d_AList, AList, sizeof(real_t*)*nodes_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      real_t alpha=1.0, beta=0.0;
      cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NSURF, 1, NSURF, &alpha, (const float**)d_BList, NSURF, (const float**)d_AList, NSURF, &beta, d_CList, NSURF, nodes_by_level_idx[i].size());
      int *d_parent_by_level_idx;
      cudaMalloc(&d_parent_by_level_idx, parent_by_level_idx[i].size() * sizeof(int));
      cudaMemcpy(d_parent_by_level_idx, &parent_by_level_idx[i][0], sizeof(int)*parent_by_level_idx[i].size(), cudaMemcpyHostToDevice);
      M2MKernel<<<parent_by_level_idx[i].size(), NSURF>>>(d_upward_equiv, d_buffer, d_parent_by_level_idx);
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      cudaFree(d_buffer);
      cudaFree(d_parent_by_level_idx);
    }
    cudaMemcpy(&upward_equiv[0], d_upward_equiv, sizeof(real_t)*upward_equiv.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_upward_equiv);
    cudaFree(d_mat_M2M);
  }

  void P2PGPU(std::vector<int> &leafs_idx, std::vector<real_t> nodes_coord, std::vector<real_t> nodes_pt_src, std::vector<int> nodes_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val, int leafs_size, int ncrit) {
    int BLOCKS = leafs_size;
    int THREADS = ncrit;

    int *d_nodes_pt_src_idx, *d_P2Plists, *d_P2Plists_idx, *d_leafs_idx;
    real_t *d_nodes_coord, *d_nodes_pt_src, *d_trg_val;
    
    cudaMalloc(&d_leafs_idx, sizeof(int)*leafs_idx.size());
    cudaMalloc(&d_nodes_pt_src_idx, sizeof(int)*nodes_pt_src_idx.size());
    cudaMalloc(&d_P2Plists, sizeof(int)*P2Plists.size());
    cudaMalloc(&d_P2Plists_idx, sizeof(int)*P2Plists_idx.size());
    cudaMalloc(&d_nodes_coord, sizeof(real_t)*nodes_coord.size());
    cudaMalloc(&d_nodes_pt_src, sizeof(real_t)*nodes_pt_src.size());
    cudaMalloc(&d_trg_val, sizeof(real_t)*trg_val.size());
    
    Profile::Tic("host to device", true);
    cudaMemcpy(d_leafs_idx, &leafs_idx[0], sizeof(int)*leafs_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src_idx, &nodes_pt_src_idx[0], sizeof(int)*nodes_pt_src_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P2Plists, &P2Plists[0], sizeof(int)*P2Plists.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P2Plists_idx, &P2Plists_idx[0], sizeof(int)*P2Plists_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_coord, &nodes_coord[0], sizeof(real_t)*nodes_coord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src, &nodes_pt_src[0], sizeof(real_t)*nodes_pt_src.size(), cudaMemcpyHostToDevice);
    Profile::Toc();
    Profile::Tic("gpu kernel", true);
    gradientP2PKernel<<<BLOCKS, THREADS>>>(d_leafs_idx, d_nodes_pt_src_idx, d_P2Plists, d_P2Plists_idx, d_nodes_coord, d_nodes_pt_src, d_trg_val);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Profile::Toc();
    Profile::Tic("device to host", true);
    cudaMemcpy(&trg_val[0], d_trg_val, sizeof(real_t)*trg_val.size(), cudaMemcpyDeviceToHost);
    Profile::Toc();
    cudaFree(d_leafs_idx);
    cudaFree(d_nodes_coord);
    cudaFree(d_nodes_pt_src);
    cudaFree(d_P2Plists_idx);
    cudaFree(d_P2Plists);
    cudaFree(d_nodes_pt_src_idx);
    cudaFree(d_trg_val);
  }

  cufftComplex *FFT_UpEquiv_GPU(int M2Lsources_idx_size, AlignedVec &up_equiv) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3_ = n1 * n1 * (n1 / 2 + 1);
    int dims[] = {n1,n1,n1};
    cufftHandle plan_up_equiv;
    cufftPlanMany(&plan_up_equiv, 3, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, M2Lsources_idx_size); // first call to cufft will always take time so this is why ill leave it out from the timing
    cufftComplex *d_up_equiv_fft;
    real_t *d_up_equiv;
    cudaMalloc(&d_up_equiv, sizeof(real_t)*up_equiv.size());
    cudaMalloc(&d_up_equiv_fft, sizeof(cufftComplex)*M2Lsources_idx_size*n3_);
    cudaMemcpy(d_up_equiv, &up_equiv[0], sizeof(real_t)*up_equiv.size(), cudaMemcpyHostToDevice);
    cufftExecR2C(plan_up_equiv, &d_up_equiv[0], &d_up_equiv_fft[0]);
    cufftDestroy(plan_up_equiv);
    cudaFree(d_up_equiv);
    return &d_up_equiv_fft[0];
  }

  std::vector<real_t> FFT_Check2Equiv_GPU(cufftComplex *d_dw_equiv_fft, int M2Ltargets_idx_size) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int dims[] = {n1,n1,n1};
    
    real_t *d_dnCheck;
    cudaMalloc(&d_dnCheck, sizeof(real_t)*M2Ltargets_idx_size*n3);
    cufftHandle plan_check_equiv;
    cufftPlanMany(&plan_check_equiv, 3, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, M2Ltargets_idx_size);
    cufftExecC2R(plan_check_equiv, &d_dw_equiv_fft[0], &d_dnCheck[0]);
    cufftDestroy(plan_check_equiv);
    std::vector<real_t> dnCheck(M2Ltargets_idx_size*n3);
    cudaMemcpy(&dnCheck[0], d_dnCheck, sizeof(real_t)*M2Ltargets_idx_size*n3, cudaMemcpyDeviceToHost); 
    cudaFree(d_dnCheck);
    cudaFree(d_dw_equiv_fft);
    return dnCheck;
  }

  cufftComplex *HadmardGPU(std::vector<int> &M2Ltargets_idx, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, cufftComplex *d_up_equiv_fft) {
    int BLOCKS = M2Ltargets_idx.size();
    int THREADS = n3_;

    int *d_M2Ltargets_idx, *d_M2LRelPos_start_idx, *d_index_in_up_equiv_fft, *d_M2LRelPoss;
    real_t *d_mat_M2L_Helper;
    cufftComplex *d_dw_equiv_fft;
    cudaMalloc(&d_M2Ltargets_idx, sizeof(int)*M2Ltargets_idx.size());
    cudaMalloc(&d_M2LRelPos_start_idx, sizeof(int)*M2LRelPos_start_idx.size());
    cudaMalloc(&d_index_in_up_equiv_fft, sizeof(int)*index_in_up_equiv_fft.size());
    cudaMalloc(&d_M2LRelPoss, sizeof(int)*M2LRelPoss.size());
    cudaMalloc(&d_dw_equiv_fft, sizeof(cufftComplex)*M2Ltargets_idx.size()*n3_);
    cudaMalloc(&d_mat_M2L_Helper, sizeof(real_t)*mat_M2L_Helper.size());

    cudaMemcpy(d_M2Ltargets_idx, &M2Ltargets_idx[0], sizeof(int)*M2Ltargets_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2LRelPos_start_idx, &M2LRelPos_start_idx[0], sizeof(int)*M2LRelPos_start_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_in_up_equiv_fft, &index_in_up_equiv_fft[0], sizeof(int)*index_in_up_equiv_fft.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2LRelPoss, &M2LRelPoss[0], sizeof(int)*M2LRelPoss.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_M2L_Helper, &mat_M2L_Helper[0], sizeof(real_t)*mat_M2L_Helper.size(), cudaMemcpyHostToDevice);    
    hadmard_kernel<<<BLOCKS, THREADS>>>(d_M2Ltargets_idx, d_up_equiv_fft, d_dw_equiv_fft, d_M2LRelPos_start_idx, d_index_in_up_equiv_fft, d_M2LRelPoss, d_mat_M2L_Helper, n3_, BLOCKS);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaFree(d_M2Ltargets_idx);
    cudaFree(d_M2LRelPos_start_idx);
    cudaFree(d_index_in_up_equiv_fft);
    cudaFree(d_M2LRelPoss);
    cudaFree(d_up_equiv_fft);
    cudaFree(d_mat_M2L_Helper);
    return &d_dw_equiv_fft[0];
  }
  
  std::vector<real_t> M2LGPU(std::vector<int> &M2Ltargets_idx, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, AlignedVec &up_equiv, int M2Lsources_idx_size) {
    cufftComplex *d_up_equiv_fft = FFT_UpEquiv_GPU(M2Lsources_idx_size, up_equiv);
    cufftComplex *d_dw_equiv_fft = HadmardGPU(M2Ltargets_idx, M2LRelPos_start_idx, index_in_up_equiv_fft, M2LRelPoss, mat_M2L_Helper, n3_, d_up_equiv_fft);
    return FFT_Check2Equiv_GPU(d_dw_equiv_fft, M2Ltargets_idx.size());
  }
}
