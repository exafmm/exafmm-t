include $(top_srcdir)/Makefile.am.include

noinst_PROGRAMS = 

# simd p2p tests
noinst_PROGRAMS += p2p_laplace p2p_helmholtz p2p_modified_helmholtz
p2p_laplace_SOURCES = p2p_laplace.cpp
p2p_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
p2p_laplace_LDADD = $(LIBS_LDADD)

p2p_helmholtz_SOURCES = p2p_helmholtz.cpp
p2p_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
p2p_helmholtz_LDADD = $(LIBS_LDADD)

p2p_modified_helmholtz_SOURCES = p2p_modified_helmholtz.cpp
p2p_modified_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
p2p_modified_helmholtz_LDADD = $(LIBS_LDADD)

# kernel tests
noinst_PROGRAMS += kernel_laplace kernel_helmholtz kernel_modified_helmholtz
kernel_laplace_SOURCES = kernel_laplace.cpp
kernel_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_laplace_LDADD = $(LIBS_LDADD)

kernel_helmholtz_SOURCES = kernel_helmholtz.cpp
kernel_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_helmholtz_LDADD = $(LIBS_LDADD)

kernel_modified_helmholtz_SOURCES = kernel_modified_helmholtz.cpp
kernel_modified_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
kernel_modified_helmholtz_LDADD = $(LIBS_LDADD)

# mpi partition tests
noinst_PROGRAMS += partition
partition_SOURCES = partition.cpp
partition_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
partition_LDADD = $(LIBS_LDADD)

# local essential tree tests
noinst_PROGRAMS += let
let_SOURCES = let.cpp
let_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
let_LDADD = $(LIBS_LDADD)

# tree tests
noinst_PROGRAMS += tree
tree_SOURCES = tree.cpp
tree_CPPFLAGS = $(AM_CPPFLAGS)
tree_LDADD = $(LIBS_LDADD)

# list tests
noinst_PROGRAMS += list
list_SOURCES = list.cpp
list_CPPFLAGS = $(AM_CPPFLAGS)

# full fmm tests
noinst_PROGRAMS += fmm_laplace fmm_laplace_non_adaptive
fmm_laplace_SOURCES = fmm_laplace.cpp
fmm_laplace_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_laplace_LDADD = $(LIBS_LDADD)

fmm_laplace_non_adaptive_SOURCES = $(fmm_laplace_SOURCES)
fmm_laplace_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_laplace_CPPFLAGS)
fmm_laplace_non_adaptive_LDADD = $(fmm_laplace_LDADD)

noinst_PROGRAMS += fmm_helmholtz fmm_helmholtz_non_adaptive
fmm_helmholtz_SOURCES = fmm_helmholtz.cpp
fmm_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_helmholtz_LDADD = $(LIBS_LDADD)

fmm_helmholtz_non_adaptive_SOURCES = $(fmm_helmholtz_SOURCES)
fmm_helmholtz_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_helmholtz_CPPFLAGS)
fmm_helmholtz_non_adaptive_LDADD = $(fmm_helmholtz_LDADD)

noinst_PROGRAMS += fmm_modified_helmholtz fmm_modified_helmholtz_non_adaptive
fmm_modified_helmholtz_SOURCES = fmm_modified_helmholtz.cpp
fmm_modified_helmholtz_CPPFLAGS = $(PRECISION) $(AM_CPPFLAGS)
fmm_modified_helmholtz_LDADD = $(LIBS_LDADD)

fmm_modified_helmholtz_non_adaptive_SOURCES = $(fmm_modified_helmholtz_SOURCES)
fmm_modified_helmholtz_non_adaptive_CPPFLAGS = -DNON_ADAPTIVE $(fmm_modified_helmholtz_CPPFLAGS)
fmm_modified_helmholtz_non_adaptive_LDADD = $(fmm_modified_helmholtz_LDADD)

check_PROGRAMS = $(noinst_PROGRAMS)
TESTS = $(noinst_PROGRAMS)
