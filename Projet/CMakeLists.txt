cmake_minimum_required(VERSION 3.18)
project("Projet MPNA" C)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_library(MATH_LIBRARY REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DOMPI_SKIP_MPICXX)
add_library(custom_lib STATIC src/io.c src/miram.c src/blas_custom.c src/tools.c)

add_executable(MIRAM src/main.c)
add_executable(NO_MPI_TESTS src/tests_no_mpi.c)

add_link_options(-fsanitize=address -fsanitize=undefined)
add_compile_options(
                      -Wall -Wextra
                      -pedantic
                      -O3
                      -fno-omit-frame-pointer
                      -g
                      -lm
                      -march=native -mtune=native -mavx2
                      -fopenmp
                      -flto
                      -fsanitize=address -fsanitize=undefined
                      -finline-functions -funroll-loops
                      -ftree-loop-vectorize -ftree-vectorize
                      -fopt-info-all=miram.gcc.optrpt
                      )

target_compile_options(custom_lib PRIVATE)
target_compile_options(MIRAM PRIVATE )
target_compile_options(NO_MPI_TESTS PRIVATE)

include_directories("include/")


target_link_libraries(custom_lib OpenMP::OpenMP_C)
target_link_libraries(custom_lib ${MATH_LIBRARY})
target_link_libraries(custom_lib ${BLAS_LIBRARIES})
target_link_libraries(MIRAM OpenMP::OpenMP_C)
target_link_libraries(MIRAM custom_lib)
target_link_libraries(MIRAM MPI::MPI_C)
target_link_libraries(MIRAM ${BLAS_LIBRARIES})
target_link_libraries(NO_MPI_TESTS OpenMP::OpenMP_C)
target_link_libraries(NO_MPI_TESTS custom_lib)
target_link_libraries(NO_MPI_TESTS ${BLAS_LIBRARIES})
find_library(MATH_LIBRARY m)


