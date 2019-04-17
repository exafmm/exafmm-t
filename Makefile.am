SUBDIRS = tests wrappers

include $(top_srcdir)/Makefile.am.include

touch:
	touch --date="`date`" aclocal.m4 Makefile.am configure Makefile.in

COMMON_SRCS = src/geometry.cpp
LAPLACE_SRCS = src/precompute_laplace.cpp src/laplace.cpp $(COMMON_SRCS)
HELMHOLTZ_SRCS = src/precompute_helmholtz.cpp src/helmholtz.cpp $(COMMON_SRCS)

bin_PROGRAMS = laplace helmholtz

laplace_SOURCES = main.cpp $(LAPLACE_SRCS)
laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
laplace_LDADD = $(LIBS_LDADD)

helmholtz_SOURCES = main.cpp $(HELMHOLTZ_SRCS)
helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)
helmholtz_LDADD = $(LIBS_LDADD)
