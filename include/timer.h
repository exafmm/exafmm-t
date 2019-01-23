#ifndef timer_h
#define timer_h
#include <map>
#include "print.h"
#include <sys/time.h>

namespace exafmm_t {
  timeval time;
  std::map<std::string,timeval> timer;

  void start(std::string event) {
    gettimeofday(&time, NULL);
    timer[event] = time;
  }

  double stop(std::string event) {
    gettimeofday(&time, NULL);
    double eventTime = time.tv_sec - timer[event].tv_sec +
      (time.tv_usec - timer[event].tv_usec) * 1e-6;
    print(event, eventTime);
    return eventTime;
  }
}
#endif
