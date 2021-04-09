#ifndef timer_h
#define timer_h
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sys/time.h>
#include <unistd.h>

namespace exafmm_t {
  static const int stringLength = 20;           //!< Length of formatted string
  static const int decimal = 7;                 //!< Decimal precision
  static const int wait = 100;                  //!< Waiting time between output of different ranks
  static const int dividerLength = stringLength + decimal + 9;  // length of output section divider
  long long flop = 0;
  timeval time;
  std::map<std::string, timeval> timer;

  void print(std::string s) {
    // if (!VERBOSE | (MPIRANK != 0)) return;
    s += " ";
    std::cout << "--- " << std::setw(stringLength) << std::left
              << std::setfill('-') << s << std::setw(decimal+1) << "-"
              << std::setfill(' ') << std::endl;
  }

  template<typename T>
  void print(std::string s, T v, bool fixed=true) {
    std::cout << std::setw(stringLength) << std::left << s << " : ";
    if(fixed)
      std::cout << std::setprecision(decimal) << std::fixed << std::scientific;
    else
      std::cout << std::setprecision(1) << std::scientific;
    std::cout << v << std::endl;
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

  void start(std::string event) {
    gettimeofday(&time, NULL);
    timer[event] = time;
  }

  double stop(std::string event, bool verbose=true) {
    gettimeofday(&time, NULL);
    double eventTime = time.tv_sec - timer[event].tv_sec +
      (time.tv_usec - timer[event].tv_usec) * 1e-6;
    if (verbose)
      print(event, eventTime);
    return eventTime;
  }
}
#endif
