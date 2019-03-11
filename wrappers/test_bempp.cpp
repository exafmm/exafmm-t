#include <iostream>
#include "exafmm_t.h"

using namespace exafmm_t;

extern "C" Bodies array_to_bodies(size_t count, real_t* coord, real_t* value, bool is_source=true);
extern "C" void init_FMM(int threads);
extern "C" Nodes setup_FMM(Bodies& sources, Bodies& targets, NodePtrs& leafs);
extern "C" void run_FMM(Nodes& nodes, NodePtrs& leafs);

int main(int argc, char **argv) {
  // generate random coordinates and charges
  size_t N = 1000000;
  real_t * src_coord = new real_t [3*N];
  real_t * src_q = new real_t [N];
  real_t * trg_coord = new real_t [3*N];
  real_t * trg_p = new real_t [N];
  for(size_t i=0; i<N; ++i) {
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    }
    src_q[i] = drand48();
  }
  Bodies sources = array_to_bodies(N, src_coord, src_q);
  Bodies targets = array_to_bodies(N, trg_coord, trg_p, false);

  // initialize global variables
  int threads = 6;
  init_FMM(threads);

  // setup FMM
  NodePtrs leafs;
  Nodes nodes = setup_FMM(sources, targets, leafs);

  // run FMM
  run_FMM(nodes, leafs);

  // delete arrays
  delete[] src_coord;
  delete[] src_q;
  delete[] trg_coord;
  delete[] trg_p;
  
  return 0;
}
