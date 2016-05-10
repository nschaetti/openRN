AC_PREREQ([2.67])
AC_INIT([libopenrn], [0.0.1], [n.schaetti@gmail.com])
AM_INIT_AUTOMAKE([1.10 -Wall no-define])
AC_CONFIG_MACRO_DIR([m4])

# Check for programs
AC_PROG_CXX
AM_PROG_AR
LT_INIT([disable-static])
AC_ENABLE_SHARED
AC_DISABLE_STATIC

# Flags
CFLAGS+="-O2 -larmadillo -std=c++11"

# Config files
AC_CONFIG_FILES([Makefile src/Makefile])

AC_OUTPUT
