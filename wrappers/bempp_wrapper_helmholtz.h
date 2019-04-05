#ifndef bempp_wrapper_helmholtz_h
#define bempp_wrapper_helmholtz_h
#include <complex>

extern "C" void init_FMM(int threads);

extern "C" void setup_FMM(int src_count, double* src_coord,
                          int trg_count, double* trg_coord);

extern "C" void run_FMM(std::complex<double>* src_value, std::complex<double>* trg_value);

extern "C" void verify_FMM(int src_count, double* src_coord, std::complex<double>* src_value,
                           int trg_count, double* trg_coord, std::complex<double>* trg_value);
#endif
