SUBDIRS = wrappers

include $(top_srcdir)/Makefile.am.include

touch:
	touch --date="`date`" aclocal.m4 Makefile.am configure Makefile.in

COMMON_SRCS = src/geometry.cpp
LAPLACE_SRCS = src/precompute_laplace.cpp src/laplace.cpp $(COMMON_SRCS)
HELMHOLTZ_SRCS = src/precompute_helmholtz.cpp src/helmholtz.cpp $(COMMON_SRCS)

bin_PROGRAMS = laplace helmholtz

laplace_SOURCES = main.cpp $(LAPLACE_SRCS)
laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)

helmholtz_SOURCES = main.cpp $(HELMHOLTZ_SRCS)
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)

bin_PROGRAMS += laplace_p2p
laplace_p2p_SOURCES = tests/p2p.cpp src/precompute_laplace.cpp src/laplace.cpp src/geometry.cpp
laplace_p2p_CPPFLAGS = $(AM_CPPFLAGS)

bin_PROGRAMS += kernel_laplace kernel_helmholtz
kernel_laplace_SOURCES = tests/kernel.cpp $(LAPLACE_SRCS)
kernel_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)

kernel_helmholtz_SOURCES = tests/kernel.cpp $(HELMHOLTZ_SRCS)
kernel_helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)
