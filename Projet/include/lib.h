#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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

// computes the eigenvalues of an hessenberg matrix
void dhseqr_(char* JOB, char* COMPZ, ui32 N, ui32 ILO, ui32 IHI, f64* H,
             ui32 LDH, f64* WR, f64* WI, f64* Z, ui32 LDZ, f64* WORK,
             ui32 LWORK, ui32* INFO);

#endif