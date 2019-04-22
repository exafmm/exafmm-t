include $(top_srcdir)/Makefile.am.include

COMMON_SRCS = ../src/geometry.cpp
LAPLACE_SRCS = ../src/precompute_laplace.cpp ../src/laplace.cpp $(COMMON_SRCS)
HELMHOLTZ_SRCS = ../src/precompute_helmholtz.cpp ../src/helmholtz.cpp $(COMMON_SRCS)

# simd p2p tests
bin_PROGRAMS = p2p_laplace p2p_helmholtz
p2p_laplace_SOURCES = p2p.cpp $(LAPLACE_SRCS) 
p2p_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
p2p_laplace_LDADD = $(LIBS_LDADD)
p2p_helmholtz_SOURCES = p2p.cpp $(HELMHOLTZ_SRCS) 
p2p_helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)
p2p_helmholtz_LDADD = $(LIBS_LDADD)

# kernel tests
bin_PROGRAMS += kernel_laplace kernel_helmholtz
kernel_laplace_SOURCES = kernel.cpp $(LAPLACE_SRCS)
kernel_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_laplace_LDADD = $(LIBS_LDADD)
kernel_helmholtz_SOURCES = kernel.cpp $(HELMHOLTZ_SRCS)
kernel_helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)
kernel_helmholtz_LDADD = $(LIBS_LDADD)

# tree tests
bin_PROGRAMS += tree
tree_SOURCES = tree.cpp
tree_CPPFLAGS = $(AM_CPPFLAGS)
