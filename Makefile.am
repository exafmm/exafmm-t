# automake preprocessor, compiler and linker flags
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CXXFLAGS = -g -O3 -mavx -fabi-version=6 -std=c++11
AM_LDFLAGS = -lfftw3 -lfftw3f -lpthread -lblas -llapack -lm

# OpenMP flags
AM_CXXFLAGS += $(OPENMP_CXXFLAGS)
AM_LDFLAGS += $(OPENMP_CXXFLAGS)

# Single/Double precision
if USE_FLOAT
AM_CPPFLAGS += -DUSE_FLOAT
endif

bin_PROGRAMS = laplace helmholtz

laplace_SOURCES = main.cpp src/precompute_laplace.cpp src/geometry.cpp src/laplace.cpp
laplace_CPPFLAGS = $(AM_CPPFLAGS)

helmholtz_SOURCES = main.cpp src/precompute_helmholtz.cpp src/geometry.cpp src/helmholtz.cpp
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(AM_CPPFLAGS)
