
AM_CPPFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11 -fopenmp -I$(top_srcdir)/include
AM_LDFLAGS = -lfftw3 -lfftw3f -lpthread -lblas -llapack -lm -fopenmp

bin_PROGRAMS = laplace

laplace_SOURCES = main.cpp src/precompute_laplace.cpp src/geometry.cpp src/laplace.cpp
laplace_CPPFLAGS = -DFLOAT $(AM_CPPFLAGS)
