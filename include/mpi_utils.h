#ifndef mpi_utils_h
#define mpi_utils_h
#include <iostream>
#include <mpi.h>

namespace exafmm_t {
#if FLOAT
  MPI_Datatype MPI_REAL_T = MPI_FLOAT;        //!< Floating point MPI type
#else
  MPI_Datatype MPI_REAL_T = MPI_DOUBLE;        //!< Floating point MPI type
#endif
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

  template<typename T>
  void printMPI(T data) {
    int size = sizeof(data);
    std::vector<T> recv(MPISIZE);
    MPI_Gather(&data, size, MPI_BYTE, &recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (MPIRANK == 0) {
      for (int irank=0; irank<MPISIZE; irank++ ) {
        std::cout << recv[irank] << " ";
      }
      std::cout << std::endl;
    }
  }
}
#endif
