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
}
