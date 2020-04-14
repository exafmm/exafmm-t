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
  extern long long flop;
  extern timeval time;
  extern std::map<std::string, timeval> timer;

  void start(std::string event);
  double stop(std::string event);
  void print(std::string s);
  void print_divider(std::string s);

  void add_flop(long long n);

  template<typename T>
  void print(std::string s, T v, bool fixed=true) {
    std::cout << std::setw(stringLength) << std::left << s << " : ";
    if(fixed)
      std::cout << std::setprecision(decimal) << std::fixed << std::scientific;
    else
      std::cout << std::setprecision(1) << std::scientific;
    std::cout << v << std::endl;
  }


}
#endif
