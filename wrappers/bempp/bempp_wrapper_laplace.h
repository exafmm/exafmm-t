#ifndef bempp_wrapper_laplace_h
#define bempp_wrapper_laplace_h

extern "C" void init_FMM(int p, int maxlevel, int threads, double wavek=20);

extern "C" void setup_FMM(int src_count, double* src_coord,
                          int trg_count, double* trg_coord);

extern "C" void run_FMM(double* src_value, double* trg_value);

extern "C" void verify_FMM(int src_count, double* src_coord, double* src_value,
                           int trg_count, double* trg_coord, double* trg_value);
#endif
