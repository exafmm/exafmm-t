#ifndef mpi_utils_h
#define mpi_utils_h
#include <mpi.h>

namespace exafmm_t {
  int MPIRANK;      //!< Rank of MPI communicator
  int MPISIZE;      //!< Size of MPI communicator
  int EXTERNAL;     //!< Flag to indicate external MPI_Init/Finalize

  void startMPI(int argc, char ** argv) {
    MPI_Initialized(&EXTERNAL);
    if (!EXTERNAL) MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  }

  void stopMPI() {
    if (!EXTERNAL) MPI_Finalize();
  }
}
#endif
