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
void potentialP2PKernel(real_t *src_coord, real_t *src_value, real_t *trg_coord, real_t *trg_value, int adj_cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int src_cnt = 64*27;
    const real_t COEFP = 1.0/(2*4*M_PI);
  const real_t COEFG = -1.0/(4*2*2*6*M_PI);
    real_t tx = trg_coord[3*i+0];
    real_t ty = trg_coord[3*i+1];
    real_t tz = trg_coord[3*i+2];
    real_t tv0=0;
      real_t tv1=0;
      real_t tv2=0;
      real_t tv3=0;
    for(int j=blockIdx.x*src_cnt; j<blockIdx.x*src_cnt+src_cnt; j++) {
    	real_t sx = src_coord[3*j+0] - tx;
        real_t sy = src_coord[3*j+1] - ty;
        real_t sz = src_coord[3*j+2] - tz;
        real_t r2 = sx*sx + sy*sy + sz*sz;
        real_t sv = src_value[j];
        if (r2 != 0)
        {
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

void P2PGPU(real_t* trg_coord, real_t* trg_value, real_t* src_coord, real_t* src_value, int leafs_cnt, int ncrit, int adj_cnt) {
int THREADS = ncrit;
    int BLOCKS = leafs_cnt;
    real_t *src_coord_device, *src_value_device, *trg_coord_device, *trg_value_device;
    
    cudaMalloc(&src_coord_device,sizeof(real_t)*leafs_cnt*ncrit*3*adj_cnt);
    cudaMalloc(&src_value_device, sizeof(real_t)*leafs_cnt*ncrit*adj_cnt);
    cudaMalloc(&trg_coord_device,sizeof(real_t)*leafs_cnt*ncrit*3);
    cudaMalloc(&trg_value_device, sizeof(real_t)*leafs_cnt*ncrit*4);
    
    cudaMemcpy(src_coord_device, src_coord, sizeof(real_t)*leafs_cnt*ncrit*3*adj_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(src_value_device, src_value, sizeof(real_t)*leafs_cnt*ncrit*adj_cnt, cudaMemcpyHostToDevice);
    cudaMemcpy(trg_coord_device, trg_coord, sizeof(real_t)*leafs_cnt*ncrit*3, cudaMemcpyHostToDevice);
    cudaMemcpy(trg_value_device, trg_value, sizeof(real_t)*leafs_cnt*4*ncrit, cudaMemcpyHostToDevice);
    potentialP2PKernel<<<BLOCKS,THREADS>>>(src_coord_device, src_value_device, trg_coord_device, trg_value_device,adj_cnt);
    cudaMemcpy(trg_value, trg_value_device, sizeof(real_t)*leafs_cnt*4*ncrit, cudaMemcpyDeviceToHost);
 
  }
}
