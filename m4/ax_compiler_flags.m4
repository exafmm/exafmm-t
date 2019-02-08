# ========================================================================================
#  Originally http://www.gnu.org/software/autoconf-archive/ax_compiler_flags_cxxflags.html
# ========================================================================================
#
# SYNOPSIS
#
#   AX_COMPILER_FLAGS()
#
# DESCRIPTION
#
#   Add warning flags for given compiler to VARIABLE, which defaults to
#   ax_compiler_c/cxx/fcflags. VARIABLE is AC_SUBST-ed by this macro,
#   but must be manually added to the CFLAGS/CXXFLAGS/FCFLAGS variable
#   for each target in the code base.
#
#   This macro depends on the environment set up by AX_COMPILER_FLAGS.
#
# LICENSE
#
#   Copyright (c) 2015 David King <amigadave@amigadave.com>
#   Copyright (c) 2016 Rio Yokota <rioyokota@gmail.com>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.  This file is offered as-is, without any
#   warranty.

#serial 8

AC_DEFUN([AX_COMPILER_FLAGS],[
    AC_REQUIRE([AC_PROG_SED])
    AX_REQUIRE_DEFINED([AX_CHECK_COMPILE_FLAG])
    # We need to turn warnings to errors for AX_APPEND_COMPILE_FLAGS to be able to work.
    # Different compilers require different flags for this:
    # GNU: -Werror
    # Intel: -diag-error warn
    # Clang: -Werror=unknown-warning-option
    AX_CHECK_COMPILE_FLAG([-Werror=unknown-warning-option],[
        ax_compiler_flags_test="-Werror=unknown-warning-option"
    ],[
        ax_compiler_flags_test="-Werror"
    ])
    AX_CHECK_COMPILE_FLAG([-diag-error warn],[
        ax_compiler_flags_test="-diag-error warn"
    ])

    # Base flags
    AX_APPEND_COMPILE_FLAGS([dnl
        -O0 dnl
        -g dnl
    ],ax_compiler_[]_AC_LANG_ABBREV[]flags,[$ax_compiler_flags_test])

    # http://stackoverflow.com/questions/3375697/useful-gcc-flags-for-c
    # http://stackoverflow.com/questions/5088460/flags-to-enable-thorough-and-verbose-g-warnings
    AX_APPEND_COMPILE_FLAGS([ dnl
        $ax_compiler_flags_test dnl
        "-check all" dnl
        "-debug all" dnl
	"-diag-disable remark" dnl
	-fmudflap dnl
	-fno-strict-aliasing dnl
	-fsanitize=address dnl
	-fsanitize=leak dnl
	-fstack-protector dnl
	-ftrapuv dnl
	-ftrapv dnl
	-traceback dnl
	dnl -Waggregate-return dnl Not compatible with idiomatic C++
	-Wall dnl
	-Warray-bounds dnl
        -Wbad-function-cast dnl
	dnl -Wconversion dnl Dealing with this will siginficantly reduce readability
	-Wcast-align dnl
	-Wcast-qual dnl
	-Wextra dnl
	-Wfatal-errors dnl
	dnl -Wfloat-equal dnl
	-Wformat=2 dnl
	-Wformat-nonliteral dnl
	-Wformat-security dnl
	-Winit-self dnl
	dnl -Winline dnl
	-Wmissing-format-attribute dnl
	-Wmissing-include-dirs dnl
	-Wmissing-noreturn dnl
        -Wnested-externs dnl
	-Wno-missing-field-initializers dnl
	-Wno-overloaded-virtual dnl
        -Wno-unused-local-typedefs dnl
	-Wno-unused-parameter dnl
        -Wno-unused-variable dnl
	dnl -Wpacked dnl Can't seem to get test_charmm.f90 to work with this
	-Wpointer-arith dnl
	-Wredundant-decls dnl
	-Wreturn-type dnl
	-Wshadow dnl
	-Wsign-compare dnl
	-Wstrict-aliasing dnl
	-Wstrict-overflow=5 dnl
	-Wstrict-prototype dnl
	dnl -Wswitch-default dnl Vectorclass doesn't have default cases
	-Wswitch-enum dnl
	dnl -Wundef dnl Requires all macros to be defined to either 0 or 1
	-Wuninitialized dnl
	-Wunreachable-code dnl
	-Wunused-but-set-variable dnl
	-Wwrite-strings dnl
    ],ax_compiler_[]_AC_LANG_ABBREV[]flags,[$ax_compiler_flags_test])

    # In the flags below, when disabling specific flags, always add *both*
    # -Wno-foo and -Wno-error=foo. This fixes the situation where (for example)
    # we enable -Werror, disable a flag, and a build bot passes C/CXX/FCFLAGS=-Wall,
    # which effectively turns that flag back on again as an error.
    for flag in $ax_compiler_[]_AC_LANG_ABBREV[]flags; do
        AS_CASE([$flag],
                [-Wno-*=*],[],
                [-Wno-*],[
                    AX_APPEND_COMPILE_FLAGS([-Wno-error=$(AS_ECHO([$flag]) | $SED 's/^-Wno-//')],
                                            ax_compiler_[]_AC_LANG_ABBREV[]flags,
                                            [$ax_compiler_flags_test])
                ])
    done

    COMPILER_[]_AC_LANG_PREFIX[]FLAGS=$ax_compiler_[]_AC_LANG_ABBREV[]flags
    # Substitute the variables
    AC_SUBST(COMPILER_[]_AC_LANG_PREFIX[]FLAGS)
    AC_SUBST(ax_compiler_[]_AC_LANG_ABBREV[]flags)
])dnl AX_COMPILER_FLAGS
