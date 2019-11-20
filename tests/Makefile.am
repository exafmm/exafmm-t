include $(top_srcdir)/Makefile.am.include

COMMON_SRCS = ../src/geometry.cpp ../src/timer.cpp
LAPLACE_SRCS = ../src/laplace.cpp $(COMMON_SRCS)
HELMHOLTZ_SRCS = ../src/helmholtz.cpp $(COMMON_SRCS)
MODIFIED_HELMHOLTZ_SRCS = ../src/modified_helmholtz.cpp $(COMMON_SRCS)

# simd p2p tests
bin_PROGRAMS = p2p_laplace p2p_helmholtz
p2p_laplace_SOURCES = p2p.cpp $(LAPLACE_SRCS) 
p2p_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
p2p_laplace_LDADD = $(LIBS_LDADD)
p2p_helmholtz_SOURCES = p2p.cpp $(HELMHOLTZ_SRCS) 
p2p_helmholtz_CPPFLAGS = -DCOMPLEX -DHELMHOLTZ $(PRECISION) $(AM_CPPFLAGS)
p2p_helmholtz_LDADD = $(LIBS_LDADD)

# kernel tests
bin_PROGRAMS += kernel_laplace kernel_helmholtz kernel_modified_helmholtz
kernel_laplace_SOURCES = kernel_laplace.cpp $(LAPLACE_SRCS)
kernel_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_laplace_LDADD = $(LIBS_LDADD)
kernel_helmholtz_SOURCES = kernel_helmholtz.cpp $(HELMHOLTZ_SRCS)
kernel_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_helmholtz_LDADD = $(LIBS_LDADD)
kernel_modified_helmholtz_SOURCES = kernel_modified_helmholtz.cpp $(MODIFIED_HELMHOLTZ_SRCS)
kernel_modified_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_modified_helmholtz_LDADD = $(LIBS_LDADD)

# tree tests
bin_PROGRAMS += tree
tree_SOURCES = tree.cpp
tree_CPPFLAGS = $(AM_CPPFLAGS)

# list tests
bin_PROGRAMS += list
list_SOURCES = list.cpp $(COMMON_SRCS)
list_CPPFLAGS = $(AM_CPPFLAGS)

# full fmm tests
bin_PROGRAMS += fmm_laplace fmm_laplace_non_adaptive
fmm_laplace_SOURCES = fmm_laplace.cpp $(LAPLACE_SRCS)
fmm_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_laplace_LDADD = $(LIBS_LDADD)

fmm_laplace_non_adaptive_SOURCES = $(fmm_laplace_SOURCES)
fmm_laplace_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_laplace_CPPFLAGS)
fmm_laplace_non_adaptive_LDADD = $(fmm_laplace_LDADD)

bin_PROGRAMS += fmm_helmholtz fmm_helmholtz_non_adaptive
fmm_helmholtz_SOURCES = fmm_helmholtz.cpp $(HELMHOLTZ_SRCS)
fmm_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_helmholtz_LDADD = $(LIBS_LDADD)

fmm_helmholtz_non_adaptive_SOURCES = $(fmm_helmholtz_SOURCES)
fmm_helmholtz_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_helmholtz_CPPFLAGS)
fmm_helmholtz_non_adaptive_LDADD = $(fmm_helmholtz_LDADD)

bin_PROGRAMS += fmm_modified_helmholtz fmm_modified_helmholtz_non_adaptive
fmm_modified_helmholtz_SOURCES = fmm_modified_helmholtz.cpp $(MODIFIED_HELMHOLTZ_SRCS)
fmm_modified_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_modified_helmholtz_LDADD = $(LIBS_LDADD)

fmm_modified_helmholtz_non_adaptive_SOURCES = $(fmm_modified_helmholtz_SOURCES)
fmm_modified_helmholtz_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_modified_helmholtz_CPPFLAGS)
fmm_modified_helmholtz_non_adaptive_LDADD = $(fmm_modified_helmholtz_LDADD)

TESTS = $(bin_PROGRAMS)
