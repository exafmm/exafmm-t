include ../../Makefile.am.include

COMMON_SRCS = ../../src/geometry.cpp
LAPLACE_SRCS = ../../src/precompute_laplace.cpp ../../src/laplace.cpp $(COMMON_SRCS)
HELMHOLTZ_SRCS = ../../src/precompute_helmholtz.cpp ../../src/helmholtz.cpp $(COMMON_SRCS)

# compile library with double precision
# append $(PRECISION) to CPPFLAGS to compile with single precision

lib_LIBRARIES = libbempp_exafmm_laplace.a libbempp_exafmm_helmholtz.a

libbempp_exafmm_laplace_a_SOURCES = bempp_wrapper.cpp $(LAPLACE_SRCS) 
libbempp_exafmm_laplace_a_CPPFLAGS = -DSORT_BACK -DNON_ADAPTIVE $(AM_CPPFLAGS)

libbempp_exafmm_helmholtz_a_SOURCES = bempp_wrapper.cpp $(HELMHOLTZ_SRCS) 
libbempp_exafmm_helmholtz_a_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ -DSORT_BACK -DNON_ADAPTIVE $(AM_CPPFLAGS)

bin_PROGRAMS = bempp_laplace bempp_helmholtz

bempp_laplace_SOURCES = bempp_example_laplace.cpp
bempp_laplace_CPPFLAGS = $(libbempp_exafmm_laplace_a_CPPFLAGS)
bempp_laplace_LDADD = libbempp_exafmm_laplace.a $(LIBS_LDADD)

bempp_helmholtz_SOURCES = bempp_example_helmholtz.cpp
bempp_helmholtz_CPPFLAGS = $(libbempp_exafmm_helmholtz_a_CPPFLAGS)
bempp_helmholtz_LDADD = libbempp_exafmm_helmholtz.a $(LIBS_LDADD)

TESTS = bempp_laplace bempp_helmholtz
