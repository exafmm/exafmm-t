#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ([2.69])
AC_INIT([exaFMM-t], [1.0], [twang66@gwu.edu])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-aux])
AC_LANG([C++])

AM_INIT_AUTOMAKE([foreign subdir-objects])
LT_INIT

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_F77

# Choose precision
AC_ARG_ENABLE([single],
              [AS_HELP_STRING([--enable-single],[compile in single precision])],
              [enable_single=$enableval],
              [enable_single=no])
AC_ARG_ENABLE([float],
              [AS_HELP_STRING([--enable-float],[synonym for --enable-single])],
              [enable_single=$enableval])
AM_CONDITIONAL([USE_FLOAT], [test "$enable_single" = "yes"])

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

# SIMD extensions, compiler flags passed to SIMD_FLAGS, CPUEXT_FLAGS
AC_ARG_ENABLE([simd],
              [AS_HELP_STRING([--disable-simd],[disable SIMD])],
              [enable_simd=$enableval],
              [enable_simd=yes])
if test "$enable_simd" = "yes"; then
   AX_EXT
fi

# Checks for libraries.
# set BLAS_LIBS, LAPACK_LIBS
AX_BLAS([], [AC_MSG_ERROR([could not find BLAS library])])
AC_SUBST([BLAS_LIBS])
AX_LAPACK([], [AC_MSG_ERROR([could not find LAPACK library])])
AC_SUBST([LAPACK_LIBS])

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T
AC_TYPE_UINT64_T

# Checks for library functions.

AC_OUTPUT
