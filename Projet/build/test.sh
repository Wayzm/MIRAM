#!/bin/bash
cmake ..
make -B
export OMP_NUM_THREADS=2
./NO_MPI_TESTS