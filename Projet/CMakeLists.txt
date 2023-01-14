cmake_minimum_required(VERSION 3.18)
project("Projet MPNA" C)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_library(custom_lib src/io.c src/miram.c src/blas_custom.c)

add_executable(MIRAM src/main.c)

add_link_options(-fsanitize=address -fsanitize=undefined)
add_compile_options(
                      -Wall -Wextra
                      -pedantic
                      -O3
                      -fno-omit-frame-pointer
                      -g
                      -march=native -mtune=native -mavx2
                      -fopenmp
                      -flto
                      -fsanitize=address -fsanitize=undefined
                      -finline-functions -funroll-loops
                      -ftree-loop-vectorize -ftree-vectorize
                      -fopt-info-all=blas.gcc.optrpt
                      )
target_compile_options(custom_lib PRIVATE)

target_compile_options(Arnoldi PRIVATE )

include_directories("include/")

target_link_libraries(custom_lib OpenMP::OpenMP_C)
target_link_libraries(custom_lib libm.a)
target_link_libraries(custom_lib ${BLAS_LIBRARIES})
target_link_libraries(MIRAM OpenMP::OpenMP_C)
target_link_libraries(MIRAM custom_lib)
target_link_libraries(MIRAM ${BLAS_LIBRARIES})