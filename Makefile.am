include $(top_srcdir)/Makefile.am.include

touch:
	touch --date="`date`" aclocal.m4 Makefile.am configure Makefile.in

bin_PROGRAMS = laplace helmholtz

laplace_SOURCES = main.cpp src/precompute_laplace.cpp src/geometry.cpp src/laplace.cpp
laplace_CPPFLAGS = $(AM_CPPFLAGS)

helmholtz_SOURCES = main.cpp src/precompute_helmholtz.cpp src/geometry.cpp src/helmholtz.cpp
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(AM_CPPFLAGS)

bin_PROGRAMS += laplace_p2p
laplace_p2p_SOURCES = tests/p2p.cpp src/precompute_laplace.cpp src/laplace.cpp src/geometry.cpp
laplace_p2p_CPPFLAGS = $(AM_CPPFLAGS)

bin_PROGRAMS += helmholtz_p2p
helmholtz_SOURCES = main.cpp src/precompute_helmholtz.cpp src/geometry.cpp src/helmholtz.cpp
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(AM_CPPFLAGS)

bin_PROGRAMS += laplace_kernel
laplace_kernel_SOURCES = tests/kernel.cpp src/precompute_laplace.cpp src/laplace.cpp src/geometry.cpp
laplace_kernel_CPPFLAGS = $(AM_CPPFLAGS)
