#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_INIT([exaFMM-t], [1.0], [twang66@gwu.edu])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_MACRO_DIR([m4])
AC_LANG([C++])
LT_INIT

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for OpenMP, compiler flags passed to OPENMP_CXXFLAGS
AC_ARG_ENABLE([openmp],
              [AS_HELP_STRING([--disable-openmp],[disable OpenMP])],
              [],
              [enable_openmp=yes])
if test "$enable_openmp" = "yes"; then
   AX_OPENMP([], [AC_MSG_ERROR([don't know how to enable OpenMP for C++])])
fi
AC_SUBST([OPENMP_CXXFLAGS])

# Checks for MPI
AC_ARG_ENABLE([mpi],
              [AS_HELP_STRING([--disable-mpi],[do not compile with MPI])],
              [],
              [enable_mpi=yes])
if test "$enable_mpi" = "yes"; then
   AX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   AC_SUBST(MPIRUN)
   CXX=$MPICXX
fi

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T
AC_TYPE_UINT64_T

# Checks for library functions.

AC_OUTPUT
