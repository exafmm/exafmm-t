include ../Makefile.am.include

lib_LIBRARIES = libbempp_exafmm.a

libbempp_exafmm_a_SOURCES = bempp.cpp ../src/precompute_laplace.cpp ../src/geometry.cpp ../src/laplace.cpp
libbempp_exafmm_a_CPPFLAGS = $(AM_CPPFLAGS)

bin_PROGRAMS = bempp

bempp_SOURCES = test_bempp.cpp
bempp_CPPFLAGS = $(libbempp_exafmm_a_CPPFLAGS)
bempp_LDADD = libbempp_exafmm.a
