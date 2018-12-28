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
  void potentialP2PKernel(int *d_leafs_idx, int *d_nodes_coord_idx, int *d_nodes_pt_src_idx, int *d_P2Plists, int *d_P2Plists_idx, real_t *d_nodes_coord, real_t *d_nodes_pt_src, real_t *d_trg_val) {

    const real_t COEFP = 1.0/(2*4*M_PI);
    const real_t COEFG = -1.0/(4*2*2*6*M_PI);

    int trg_idx = d_leafs_idx[blockIdx.x];
    int first_trg_coord_idx = d_nodes_coord_idx[trg_idx];
    int trg_coord_size = d_nodes_coord_idx[trg_idx+1] - d_nodes_coord_idx[trg_idx];
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
        int first_src_coord_idx = d_nodes_coord_idx[src_idx];
        int src_coord_size = d_nodes_coord_idx[src_idx+1] - d_nodes_coord_idx[src_idx];
        int first_src_val_idx = d_nodes_pt_src_idx[src_idx];
        for(int k=0; k<src_coord_size/3; k ++) {
          real_t sx = d_nodes_coord[first_src_coord_idx + 3*k + 0] - tx;
          real_t sy = d_nodes_coord[first_src_coord_idx + 3*k + 1] - ty;
          real_t sz = d_nodes_coord[first_src_coord_idx + 3*k + 2] - tz;
          real_t r2 = sx*sx + sy*sy + sz*sz;
          real_t sv = d_nodes_pt_src[first_src_val_idx+k];
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
  void hadmard_kernel(int *d_M2Ltargets_idx, cufftComplex *d_up_equiv_fft, real_t *d_dw_equiv_fft, int *d_M2LRelPos_start_idx, int *d_index_in_up_equiv_fft, int *d_M2LRelPoss, real_t *d_mat_M2L_Helper, int n3_, int BLOCKS) {
    int i = blockIdx.x;
    int k = threadIdx.x;
    int M2LRelPos_size = d_M2LRelPos_start_idx[i+1]-d_M2LRelPos_start_idx[i];
    for(int j=0; j <M2LRelPos_size; j++) {
      int relPosidx = d_M2LRelPoss[d_M2LRelPos_start_idx[i]+j];
      real_t *kernel = &d_mat_M2L_Helper[relPosidx*2*n3_];
      cufftComplex *equiv = &d_up_equiv_fft[d_index_in_up_equiv_fft[d_M2LRelPos_start_idx[i]+j]*n3_];
      real_t *check = &d_dw_equiv_fft[i*2*n3_];
      int real = 2*k+0;
      int imag = 2*k+1;
      check[real] += kernel[real]*equiv[k].x - kernel[imag]*equiv[k].y;
      check[imag] += kernel[real]*equiv[k].y + kernel[imag]*equiv[k].x;
    }
  }

  void cuda_init_drivers() {
    cudaFree(0);
}

  void P2PGPU(std::vector<int> leafs_idx, std::vector<real_t> nodes_coord, std::vector<int> nodes_coord_idx, std::vector<real_t> nodes_pt_src, std::vector<int> nodes_pt_src_idx, std::vector<int> P2Plists, std::vector<int> P2Plists_idx, std::vector<real_t> &trg_val) {
    int BLOCKS = leafs_idx.size();
    int THREADS = 64;

    int *d_leafs_idx, *d_nodes_coord_idx, *d_nodes_pt_src_idx, *d_P2Plists, *d_P2Plists_idx;
    real_t *d_nodes_coord, *d_nodes_pt_src, *d_trg_val;
    
    cudaMalloc(&d_leafs_idx, sizeof(int)*leafs_idx.size());
    cudaMalloc(&d_nodes_coord_idx, sizeof(int)*nodes_coord_idx.size());
    cudaMalloc(&d_nodes_pt_src_idx, sizeof(int)*nodes_pt_src_idx.size());
    cudaMalloc(&d_P2Plists, sizeof(int)*P2Plists.size());
    cudaMalloc(&d_P2Plists_idx, sizeof(int)*P2Plists_idx.size());
    cudaMalloc(&d_nodes_coord, sizeof(real_t)*nodes_coord.size());
    cudaMalloc(&d_nodes_pt_src, sizeof(real_t)*nodes_pt_src.size());
    cudaMalloc(&d_trg_val, sizeof(real_t)*trg_val.size());
    
    Profile::Tic("memcpy host to device", true);
    cudaMemcpy(d_leafs_idx, &leafs_idx[0], sizeof(int)*leafs_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_coord_idx, &nodes_coord_idx[0], sizeof(int)*nodes_coord_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src_idx, &nodes_pt_src_idx[0], sizeof(int)*nodes_pt_src_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P2Plists, &P2Plists[0], sizeof(int)*P2Plists.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P2Plists_idx, &P2Plists_idx[0], sizeof(int)*P2Plists_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_coord, &nodes_coord[0], sizeof(real_t)*nodes_coord.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes_pt_src, &nodes_pt_src[0], sizeof(real_t)*nodes_pt_src.size(), cudaMemcpyHostToDevice);
    Profile::Toc();
    Profile::Tic("gpu kernel", true);
    potentialP2PKernel<<<BLOCKS, THREADS>>>(d_leafs_idx, d_nodes_coord_idx, d_nodes_pt_src_idx, d_P2Plists, d_P2Plists_idx, d_nodes_coord, d_nodes_pt_src, d_trg_val);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Profile::Toc();
    Profile::Tic("memcpy device to host", true);
    cudaMemcpy(&trg_val[0], d_trg_val, sizeof(real_t)*trg_val.size(), cudaMemcpyDeviceToHost);
    Profile::Toc();
    cudaFree(d_nodes_coord);
    cudaFree(d_nodes_pt_src);
    cudaFree(d_P2Plists_idx);
    cudaFree(d_P2Plists);
    cudaFree(d_nodes_pt_src_idx);
    cudaFree(d_nodes_coord_idx);
    cudaFree(d_leafs_idx);
    cudaFree(d_trg_val);
  }

  void M2LGPU(std::vector<int> &M2Ltargets_idx, AlignedVec &dw_equiv_fft, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, AlignedVec &up_equiv, int M2Lsources_idx_size) {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
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
    HadmardGPU(M2Ltargets_idx, dw_equiv_fft, M2LRelPos_start_idx, index_in_up_equiv_fft, M2LRelPoss, mat_M2L_Helper, n3_, d_up_equiv_fft);
 
 }

  void HadmardGPU(std::vector<int> &M2Ltargets_idx, AlignedVec &dw_equiv_fft, std::vector<int> &M2LRelPos_start_idx, std::vector<int> &index_in_up_equiv_fft, std::vector<int> &M2LRelPoss, RealVec mat_M2L_Helper, int n3_, cufftComplex *d_up_equiv_fft) {
    int BLOCKS = M2Ltargets_idx.size();
    int THREADS = n3_;

    int *d_M2Ltargets_idx, *d_M2LRelPos_start_idx, *d_index_in_up_equiv_fft, *d_M2LRelPoss;
    real_t *d_dw_equiv_fft, *d_mat_M2L_Helper;
    
    cudaMalloc(&d_M2Ltargets_idx, sizeof(int)*M2Ltargets_idx.size());
    cudaMalloc(&d_M2LRelPos_start_idx, sizeof(int)*M2LRelPos_start_idx.size());
    cudaMalloc(&d_index_in_up_equiv_fft, sizeof(int)*index_in_up_equiv_fft.size());
    cudaMalloc(&d_M2LRelPoss, sizeof(int)*M2LRelPoss.size());
    cudaMalloc(&d_dw_equiv_fft, sizeof(real_t)*dw_equiv_fft.size());
    cudaMalloc(&d_mat_M2L_Helper, sizeof(real_t)*mat_M2L_Helper.size());

    cudaMemcpy(d_M2Ltargets_idx, &M2Ltargets_idx[0], sizeof(int)*M2Ltargets_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2LRelPos_start_idx, &M2LRelPos_start_idx[0], sizeof(int)*M2LRelPos_start_idx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_in_up_equiv_fft, &index_in_up_equiv_fft[0], sizeof(int)*index_in_up_equiv_fft.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2LRelPoss, &M2LRelPoss[0], sizeof(int)*M2LRelPoss.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dw_equiv_fft, &dw_equiv_fft[0], sizeof(real_t)*dw_equiv_fft.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_M2L_Helper, &mat_M2L_Helper[0], sizeof(real_t)*mat_M2L_Helper.size(), cudaMemcpyHostToDevice);    
    hadmard_kernel<<<BLOCKS, THREADS>>>(d_M2Ltargets_idx, d_up_equiv_fft, d_dw_equiv_fft, d_M2LRelPos_start_idx, d_index_in_up_equiv_fft, d_M2LRelPoss, d_mat_M2L_Helper, n3_, BLOCKS);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&dw_equiv_fft[0], d_dw_equiv_fft, sizeof(real_t)*dw_equiv_fft.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_M2Ltargets_idx);
    cudaFree(d_M2LRelPos_start_idx);
    cudaFree(d_index_in_up_equiv_fft);
    cudaFree(d_M2LRelPoss);
    cudaFree(d_up_equiv_fft);
    cudaFree(d_dw_equiv_fft);
    cudaFree(d_mat_M2L_Helper);
  }
}
