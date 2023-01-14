#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <cblas.h>

#define ui32 unsigned int
#define f64 double

// LU decomoposition of a general matrix
void dgetrf_(ui32* M, ui32 *N, f64* A, ui32* lda, ui32* IPIV, ui32* INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(ui32* N, f64* A, ui32* lda, ui32* IPIV, f64* WORK, ui32* lwork, ui32* INFO);

#endif