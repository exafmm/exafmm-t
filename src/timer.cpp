#include "timer.h"

namespace exafmm_t {
  timeval time;
  std::map<std::string, timeval> timer;
  long long flop = 0;

  void start(std::string event) {
    gettimeofday(&time, NULL);
    timer[event] = time;
  }

  double stop(std::string event, bool verbose) {
    gettimeofday(&time, NULL);
    double eventTime = time.tv_sec - timer[event].tv_sec +
      (time.tv_usec - timer[event].tv_usec) * 1e-6;
    if (verbose)
      print(event, eventTime);
    return eventTime;
  }

  void print(std::string s) {
    // if (!VERBOSE | (MPIRANK != 0)) return;
    s += " ";
    std::cout << "--- " << std::setw(stringLength) << std::left
              << std::setfill('-') << s << std::setw(decimal+1) << "-"
              << std::setfill(' ') << std::endl;
  }

  void print_divider(std::string s) {
    s.insert(0, " ");
    s.append(" ");
    int halfLength = (dividerLength - s.length()) / 2;
    std::cout << std::string(halfLength, '-') << s
              << std::string(dividerLength-halfLength-s.length(), '-') << std::endl;
  }
  
  void add_flop(long long n) {
#pragma omp atomic update
    flop += n;
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
