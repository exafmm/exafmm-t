#ifndef profile_h
#define profile_h
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stack>
#include <vector>
namespace exafmm_t {
class Profile {
 public:

  static long long Add_FLOP(long long inc);
  static long long Add_MEM(long long inc); 

  static bool Enable(bool state);
  static void Tic(const char* name_, bool sync_=false, int verbose=0);
  static void Toc();
  static void print();
  static void reset();
private:

  static long long FLOP;
  static long long MEM;
  static bool enable_state;
  static std::stack<bool> sync;
  static std::stack<std::string> name;
  static std::vector<long long> max_mem;

  static unsigned int enable_depth;
  static std::stack<int> verb_level;

  static std::vector<bool> e_log;
  static std::vector<bool> s_log;
  static std::vector<std::string> n_log;
  static std::vector<double> t_log;
  static std::vector<long long> f_log;
  static std::vector<long long> m_log;
  static std::vector<long long> max_m_log;
  static int test;
};

}//end namespace
#endif
