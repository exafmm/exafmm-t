#ifndef print_h
#define print_h
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace exafmm_t {
  bool VERBOSE = true;                          //!< Print to screen
  static const int stringLength = 20;           //!< Length of formatted string
  static const int decimal = 7;                 //!< Decimal precision
  static const int wait = 100;                  //!< Waiting time between output of different ranks

  void print(std::string s) {
    // if (!VERBOSE | (MPIRANK != 0)) return;
    s += " ";
    std::cout << "--- " << std::setw(stringLength) << std::left
              << std::setfill('-') << s << std::setw(decimal+1) << "-"
              << std::setfill(' ') << std::endl;
  }

  template<typename T>
  void print(std::string s, T v, bool fixed=true) {
    // if (!VERBOSE | (MPIRANK != 0)) return;
    std::cout << std::setw(stringLength) << std::left << s << " : ";
    if(fixed)
      std::cout << std::setprecision(decimal) << std::fixed;
    else
      std::cout << std::setprecision(1) << std::scientific;
    std::cout << v << std::endl;
  }

  // template<typename T>
  // void printMPI(T data) {
    // if (!VERBOSE) return;
    // int size = sizeof(data);
    // std::vector<T> recv(MPISIZE);
    // MPI_Gather(&data, size, MPI_BYTE, &recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
    // if (MPIRANK == 0) {
      // for (int irank=0; irank<MPISIZE; irank++ ) {
        // std::cout << recv[irank] << " ";
      // }
      // std::cout << std::endl;
    // }
  // }

  // template<typename T>
  // void printMPI(T data, const int irank) {
    // if (!VERBOSE) return;
    // int size = sizeof(data);
    // if (MPIRANK == irank) MPI_Send(&data, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    // if (MPIRANK == 0) {
      // MPI_Recv(&data, size, MPI_BYTE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // std::cout << data << std::endl;
    // }
  // }

  // template<typename T>
  // void printMPI(T * data, const int begin, const int end) {
    // if (!VERBOSE) return;
    // int range = end - begin;
    // int size = sizeof(*data) * range;
    // std::vector<T> recv(MPISIZE * range);
    // MPI_Gather(&data[begin], size, MPI_BYTE, &recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
    // if (MPIRANK == 0) {
      // int ic = 0;
      // for (int irank=0; irank<MPISIZE; irank++ ) {
        // std::cout << irank << " : ";
        // for (int i=0; i<range; i++, ic++) {
          // std::cout << recv[ic] << " ";
        // }
        // std::cout << std::endl;
      // }
    // }
  // }

  // template<typename T>
  // void printMPI(T * data, const int begin, const int end, const int irank) {
    // if (!VERBOSE) return;
    // int range = end - begin;
    // int size = sizeof(*data) * range;
    // std::vector<T> recv(range);
    // if (MPIRANK == irank) MPI_Send(&data[begin], size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    // if (MPIRANK == 0) {
      // MPI_Recv(&recv[0], size, MPI_BYTE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // for (int i=0; i<range; i++) {
        // std::cout << recv[i] << " ";
      // }
      // std::cout << std::endl;
    // }
  // }
}
#endif
