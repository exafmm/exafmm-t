include ../Makefile.am.include

# compile library with double precision
# append $(PRECISION) to CPPFLAGS to compile with single precision

lib_LIBRARIES = libbempp_exafmm_laplace.a libbempp_exafmm_helmholtz.a

libbempp_exafmm_laplace_a_SOURCES = bempp_wrapper.cpp ../src/precompute_laplace.cpp ../src/geometry.cpp ../src/laplace.cpp
libbempp_exafmm_laplace_a_CPPFLAGS = -DSORT_BACK $(AM_CPPFLAGS)

libbempp_exafmm_helmholtz_a_SOURCES = bempp_wrapper.cpp ../src/precompute_helmholtz.cpp ../src/geometry.cpp ../src/helmholtz.cpp
libbempp_exafmm_helmholtz_a_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ -DSORT_BACK $(AM_CPPFLAGS)

bin_PROGRAMS = bempp_laplace bempp_helmholtz

bempp_laplace_SOURCES = bempp_example_laplace.cpp
bempp_laplace_CPPFLAGS = $(libbempp_exafmm_laplace_a_CPPFLAGS)
bempp_laplace_LDADD = libbempp_exafmm_laplace.a

bempp_helmholtz_SOURCES = bempp_example_helmholtz.cpp
bempp_helmholtz_CPPFLAGS = $(libbempp_exafmm_helmholtz_a_CPPFLAGS)
bempp_helmholtz_LDADD = libbempp_exafmm_helmholtz.a
