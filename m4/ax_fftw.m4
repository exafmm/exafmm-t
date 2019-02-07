#
# SYNOPSIS
#
#   AX_FFTW([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for a version of FFTW3.  The FFTW_CPPFLAGS and FFTW
#   output variables hold the compile and link flags.
#
#   To link an application with FFTW, you should link with:
#
#       $FFTW_LIBS
#       dnl $FFTW $PTHREAD_LIBS
#
#   The user may use:
# 
#       --with-fftw=<dir>
#
#   to manually specify the installed prefix of FFTW.
#
#   ACTION-IF-FOUND is a list of shell commands to run if the FFTW library is
#   found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it is
#   not found. If ACTION-IF-FOUND is not specified, the default action will
#   define HAVE_FFTW and set output variables above.
#
#   This macro requires autoconf 2.50 or later.
#
# LAST MODIFICATION
#
#   2016-11-15
#
# COPYING
#
#   Copyright (c) 2016 Theodore Kisner <tskisner@lbl.gov>
#   Copyright (c) 2019 Tingyu Wang <twang66@gwu.edu>
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification,
#   are permitted provided that the following conditions are met:
#
#   o  Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#
#   o  Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
#   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
#   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
#   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
#   OF THE POSSIBILITY OF SUCH DAMAGE.


AC_DEFUN([AX_FFTW], [
AC_PREREQ(2.50)
dnl AC_REQUIRE([AX_PTHREAD])

acx_fftwf_ok=no
acx_fftw_ok=no
acx_fftw_threads=no
acx_fftw_ldflags=""
acx_fftw_libs="-lfftw3f -lfftw3"
acx_fftw_libs_threads="-lfftw3_threads"

FFTW_CPPFLAGS=""
FFTW=""

AC_ARG_WITH(fftw, [AC_HELP_STRING([--with-fftw=<dir>], [use FFTW installed in prefix <dir>.])])

if test x"$with_fftw" != x; then
   if test x"$with_fftw" != xno; then
      FFTW_CPPFLAGS="-I$with_fftw/include"
      acx_fftw_ldflags="-L$with_fftw/lib"
   else
      acx_fftw_ok=disable
   fi
fi

if test $acx_fftw_ok = disable; then
   echo "**** FFTW explicitly disabled by configure."
   AC_MSG_ERROR([cannot disable fftw, exafmm-t requires fftw to compile])
else

   # Save environment

   acx_fftw_save_CPPFLAGS="$CPPFLAGS"
   acx_fftw_save_LIBS="$LIBS"

   # Test serial compile and linking

   # First, try user-specified location or linked-by-default (for example using
   # a compiler wrapper script)

   FFTW="$acx_fftw_ldflags $acx_fftw_libs"
   CPPFLAGS="$CPPFLAGS $FFTW_CPPFLAGS"
   LIBS="$FFTW $acx_fftw_save_LIBS"

   AC_CHECK_HEADERS([fftw3.h])

   AC_MSG_CHECKING([for fftwf_malloc in user specified location])
   AC_TRY_LINK_FUNC(fftwf_malloc, [acx_fftwf_ok=yes;AC_DEFINE(HAVE_FFTW_F,1,[Define if you have the FFTW library in single precision.])], [])
   AC_MSG_RESULT($acx_fftwf_ok)

   AC_MSG_CHECKING([for fftw_malloc in user specified location])
   AC_TRY_LINK_FUNC(fftw_malloc, [acx_fftw_ok=yes;AC_DEFINE(HAVE_FFTW,1,[Define if you have the FFTW library in double precision.])], [])
   AC_MSG_RESULT($acx_fftw_ok)

   if test $acx_fftw_ok = yes && test $acx_fftwf_ok = yes; then
      dnl FFTW="$acx_fftw_ldflags $acx_fftw_libs_threads $acx_fftw_libs"
      FFTW="$acx_fftw_ldflags $acx_fftw_libs"
      CPPFLAGS="$CPPFLAGS $PTHREAD_CFLAGS"
      LIBS="$FFTW $acx_fftw_save_LIBS $PTHREAD_LIBS"
      
      acx_fftw_threads_ok=no;
      AC_MSG_CHECKING([for fftw_plan_with_nthreads in user specified location])
      AC_TRY_LINK_FUNC(fftw_plan_with_nthreads, [acx_fftw_threads_ok=yes;AC_DEFINE(HAVE_FFTW_THREADS,1,[Define if you have the FFTW threads library.])], [])
      AC_MSG_RESULT($acx_fftw_threads_ok)
   else
      if test $acx_fftwf_ok = no; then
         AC_MSG_ERROR([cannot find fftwf library.])
      else
         AC_MSG_ERROR([cannot find fftw library.])
      fi
   fi

   # Restore environment
   LIBS="$acx_fftw_save_LIBS"
   CPPFLAGS="$acx_fftw_save_CPPFLAGS"

fi

# Define exported variables
FFTW_LIBS="$FFTW"
AC_SUBST(FFTW_CPPFLAGS)
AC_SUBST(FFTW_LIBS)
])
