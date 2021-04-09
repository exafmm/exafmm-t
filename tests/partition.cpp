#include <numeric>    // std::accumulate
#include "dataset.h"
#include "exafmm_t.h"
#include "partition.h"
#include "timer.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  startMPI(argc, argv);

  int n = args.numBodies;
  Bodies<real_t> sources = init_sources<real_t>(n, args.distribution, MPIRANK);
  Bodies<real_t> targets = init_targets<real_t>(n, args.distribution, MPIRANK+10);
  
  vec3 x0;
  real_t r0;
  allreduceBounds(sources, targets, x0, r0);
  std::vector<int> offset;
  partition(sources, targets, x0, r0, offset, args.maxlevel);

  // print partition information
  int nsrcs = sources.size();
  int ntrgs = targets.size();
  if (MPIRANK == 0) std::cout << "number of sources: " << std::endl;
  printMPI(nsrcs);
  if (MPIRANK == 0) std::cout << "number of targets: " << std::endl;
  printMPI(ntrgs);

  int nsrcs_total, ntrgs_total;
  MPI_Reduce(&nsrcs, &nsrcs_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ntrgs, &ntrgs_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (MPIRANK == 0) {
    assert(nsrcs_total == n * MPISIZE);
    assert(ntrgs_total == n * MPISIZE);
    std::cout << "assertion passed!" << std::endl;
    std::cout << "r0: " << r0 << std::endl;
    std::cout << "x0: " << x0 << std::endl;
  }
  stopMPI();
}
