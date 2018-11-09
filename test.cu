#include <iostream>
#include <vector>

struct type {
  int a;
  float b;
};

__global__ void kernel(type* vec) {
  int i = threadIdx.x;
  printf("%d %f\n",vec[i].a,vec[i].b);
}

int main() {
  std::vector<type> vec(10);
  for (int i=0; i<10; i++) {
    vec[i].a = i;
    vec[i].b = i * 0.1;
  }
  type * d_vec;
  cudaMalloc(&d_vec, 10 * sizeof(type));
  cudaMemcpy(d_vec, &vec[0], 10 * sizeof(type), cudaMemcpyHostToDevice);
  kernel<<<1,10>>>(d_vec);
  cudaFree(d_vec);
}