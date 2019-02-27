include ../Makefile.am.include

lib_LIBRARIES = libbempp_exafmm_laplace.a libbempp_exafmm_helmholtz.a

libbempp_exafmm_laplace_a_SOURCES = bempp.cpp ../src/precompute_laplace.cpp ../src/geometry.cpp ../src/laplace.cpp
libbempp_exafmm_laplace_a_CPPFLAGS = $(AM_CPPFLAGS)

libbempp_exafmm_helmholtz_a_SOURCES = bempp.cpp ../src/precompute_helmholtz.cpp ../src/geometry.cpp ../src/helmholtz.cpp
libbempp_exafmm_helmholtz_a_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(AM_CPPFLAGS)

bin_PROGRAMS = bempp_laplace bempp_helmholtz

bempp_laplace_SOURCES = test_bempp.cpp
bempp_laplace_CPPFLAGS = $(libbempp_exafmm_laplace_a_CPPFLAGS)
bempp_laplace_LDADD = libbempp_exafmm_laplace.a

bempp_helmholtz_SOURCES = test_bempp.cpp
bempp_helmholtz_CPPFLAGS = $(libbempp_exafmm_helmholtz_a_CPPFLAGS)
bempp_helmholtz_LDADD = libbempp_exafmm_helmholtz.a
