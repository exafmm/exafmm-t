# force level 3 optimization (-O2 by default)
CXXFLAGS += -O3

# automake preprocessor, compiler and linker flags
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CXXFLAGS = -fabi-version=6 -std=c++11
AM_LDFLAGS = -lfftw3 -lfftw3f -lpthread -lm

# OpenMP flags
AM_CXXFLAGS += $(OPENMP_CXXFLAGS)
AM_LDFLAGS += $(OPENMP_CXXFLAGS)

# SIMD and other extensions flags
AM_CXXFLAGS += $(SIMD_FLAGS) $(CPUEXT_FLAGS)

# Single/Double precision
if USE_FLOAT
AM_CPPFLAGS += -DUSE_FLOAT
endif

# libraries
AM_LDFLAGS += $(BLAS_LIBS) $(LAPACK_LIBS)

bin_PROGRAMS = laplace helmholtz

laplace_SOURCES = main.cpp src/precompute_laplace.cpp src/geometry.cpp src/laplace.cpp
laplace_CPPFLAGS = $(AM_CPPFLAGS)

helmholtz_SOURCES = main.cpp src/precompute_helmholtz.cpp src/geometry.cpp src/helmholtz.cpp
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(AM_CPPFLAGS)
