#include "exafmm_t.h"
#include "laplace_cuda.h"
#include <iostream>

namespace exafmm_t{

__global__ 
void potentialP2PKernel(real_t *src_coord, real_t *src_value, real_t *trg_coord, real_t *trg_value, int src_cnt)
{
    const real_t COEF = 1.0/(2*4*M_PI);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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

 void potentialP2PGPU(RealVec& src_coord, RealVec& src_value, RealVec& trg_coord, RealVec& trg_value)
 {
   real_t *src_coord_device, *src_value_device, *trg_coord_device, *trg_value_device;

	real_t *src_coord_host = &src_coord[0], *src_value_host = &src_value[0], *trg_coord_host = &trg_coord[0], *trg_value_host = &trg_value[0];

	int trg_cnt = trg_coord.size() / 3;
	int src_cnt = src_coord.size() / 3;
	
	int threads = 1;

	cudaMalloc(&src_coord_device,sizeof(real_t)*src_coord.size());
	cudaMalloc(&src_value_device,sizeof(real_t)*src_value.size());
	cudaMalloc(&trg_coord_device,sizeof(real_t)*trg_coord.size());
	cudaMalloc(&trg_value_device, sizeof(real_t)*trg_value.size());

	cudaMemcpy(src_coord_device, src_coord_host, sizeof(real_t)*src_coord.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(src_value_device, src_value_host, sizeof(real_t)*src_value.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(trg_coord_device, trg_coord_host, sizeof(real_t)*trg_coord.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(trg_value_device, trg_value_host, sizeof(real_t)*trg_value.size(), cudaMemcpyHostToDevice);

	potentialP2PKernel<<<trg_cnt/threads,threads>>>(src_coord_device, src_value_device, trg_coord_device, trg_value_device, src_cnt);
	cudaMemcpy(trg_value_host, trg_value_device, sizeof(real_t)*trg_value.size(), cudaMemcpyDeviceToHost);
 }
}
