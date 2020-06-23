cmake_minimum_required (VERSION 3.6)
project (exafmm-t CXX)

# cmake settings
set (CMAKE_VERBOSE_MAKEFILE on)
set (CMAKE_COLOR_MAKEFILE on)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake-modules")

# set CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

# check compiler support c++11 standard
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# check openmp
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
  message (FATAL_ERROR "The compiler does not support OpenMP.")
endif()

# check BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(FFTW REQUIRED)

# targets
include_directories (${PROJECT_SOURCE_DIR}/include)

add_executable (fmm_laplace tests/fmm_laplace.cpp)
add_executable (fmm_helmholtz tests/fmm_helmholtz.cpp)
add_executable (fmm_modified_helmholtz tests/fmm_modified_helmholtz.cpp)

target_link_libraries (fmm_laplace PRIVATE ${BLAS_LIBRARIES}
                                   PRIVATE ${LAPACK_LIBRARIES}
                                   PRIVATE ${FFTW_LIBRARIES})

target_link_libraries (fmm_helmholtz PRIVATE ${BLAS_LIBRARIES}
                                     PRIVATE ${LAPACK_LIBRARIES}
                                     PRIVATE ${FFTW_LIBRARIES})

target_link_libraries (fmm_modified_helmholtz PRIVATE ${BLAS_LIBRARIES}
                                              PRIVATE ${LAPACK_LIBRARIES}
                                              PRIVATE ${FFTW_LIBRARIES})
