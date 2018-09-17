#include "build_tree.h"
#include "dataset.h"
#include "interaction_list.h"
#include "laplace_c.h"
#include "precompute_c.h"
#include "profile.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  omp_set_num_threads(args.threads);
  size_t N = args.numBodies;
  MULTIPOLE_ORDER = args.P;
  NSURF = 6*(MULTIPOLE_ORDER-1)*(MULTIPOLE_ORDER-1) + 2;
  Profile::Enable(true);
  
  int n = 20;
  RealVec src_coord(3*n), trg_coord(3*n);
  ComplexVec src_value(n), trg_value(n, complex_t(0.,0.));
  srand48(10);

  for(int i=0; i<n; ++i) {
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    } 
    src_value[i] = complex_t(drand48(), drand48());
  }

  potentialP2P(src_coord, src_value, trg_coord, trg_value);
  for(int i=0; i<n; ++i) {
    std::cout << trg_value[i] << std::endl;
  }
  return 0;
}
