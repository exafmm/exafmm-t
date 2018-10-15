#include "exafmm_t.h"
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
void potentialP2PKernel(real_t *src_coord, real_t *src_value, real_t *trg_coord, real_t *trg_value, int src_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;    
    const real_t COEF = 1.0/(2*4*M_PI);
    real_t tx = trg_coord[3*i+0];
    real_t ty = trg_coord[3*i+1];
    real_t tz = trg_coord[3*i+2];
    real_t tv = 0;
    for(int j=0; j<src_cnt; j++) {
    	real_t sx = src_coord[3*j+0] - tx;
        real_t sy = src_coord[3*j+1] - ty;
        real_t sz = src_coord[3*j+2] - tz;
        real_t r2 = sx*sx + sy*sy + sz*sz;
        real_t sv = src_value[j];
        if (r2 != 0)
        {
          real_t invR = 1.0/sqrt(r2);
          tv += invR*sv;
        }
    }
    tv *= COEF;
    trg_value[i] += tv;
}

void P2PGPU(real_t* trg_coord, real_t* trg_val, real_t* src_coord, real_t* src_val, int leafs_cnt, int ncrit, int adj_cnt) {
    int THREADS = ncrit;
    int BLOCKS = leafs_cnt;
    real_t *src_coord_device, *src_value_device, *trg_coord_device, *trg_value_device;
    
    cudaMalloc(&src_coord_device,sizeof(real_t)*leafs_cnt*ncrit*3*adj_cnt);
    cudaMalloc(&src_value_device, sizeof(real_t)*leafs_cnt*ncrit*adj_cnt);
    cudaMalloc(&trg_coord_device,sizeof(real_t)*leafs_cnt*ncrit*3);
    cudaMalloc(&trg_value_device, sizeof(real_t)*leafs_cnt*ncrit);
    
    cudaMemcpy(src_coord_device, src_coord, sizeof(real_t)*leafs_cnt*ncrit*3*adj_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(src_value_device, src_val, sizeof(real_t)*leafs_cnt*ncrit*adj_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(trg_coord_device, trg_coord, sizeof(real_t)*leafs_cnt*ncrit*3, cudaMemcpyHostToDevice);
    cudaMemcpy(trg_value_device, trg_val, sizeof(real_t)*leafs_cnt*ncrit, cudaMemcpyHostToDevice);
    potentialP2PKernel<<<BLOCKS,THREADS>>>(src_coord_device, src_value_device, trg_coord_device, trg_value_device,ncrit*adj_cnt*3);
    cudaMemcpy(trg_val, trg_value_device, sizeof(real_t)*leafs_cnt*ncrit, cudaMemcpyDeviceToHost);
 
  }
}
