#ifndef BLAS_CuSTOM_H
#define BLAS_CUSTOM_H

#include "lib.h"

/***
 * GEVV_CLASSIC computes a normal vector * vector : a = b * x * y
 * GEMV_CLASSIC computes a normal matrix * vector : z = a * M * x
 * GEMV_MODIFIED computes the nth column of the matrix * vector : a = b * M[n + cols] * x
 * GEMM_MODIFIED computes the nth column of the matrix A  and  matrix B : x = a * A[n + cols] * B[n + cols]
 ***/
f64 GEVV_CLASSIC(const ui32 size,
                 const f64 factor,
				 const f64* vecteur_x,
			     const f64* vecteur_y);

f64* GEMV_CLASSIC(const ui32 rows,
                  const ui32 cols,
                  const f64 factor,
                  const f64* __restrict__ matrix,
                  const ui32 size_vec,
                  const f64* __restrict__ vector);

f64 GEMV_MODIFIED(const ui32 rows,
                  const ui32 cols,
                  const f64 factor,
                  const ui32 increment,
                  const f64* __restrict__ matrix,
                  const ui32 size_vec,
                  const f64* __restrict__ vector);

f64* GEMM_MODIFIED(const ui32 rows_A,
                   const ui32 cols_A,
                   const f64 factor,
                   const ui32 increment_A,
                   const f64* __restrict__ matrix_A,
                   const ui32 rows_B,
                   const ui32 cols_B,
                   const f64* __restrict__ matrix_B);

void GEMM_CLASSIC(const ui32 rows_A,
                  const ui32 cols_A,
                  const f64 factor,
                  const f64* __restrict__ matrix_A,
                  const ui32 rows_B,
                  const ui32 cols_B,
                  const f64* __restrict__ matrix_B,
                  f64* __restrict__ matrix_C);

void GEMM_CLASSIC_NO_C(const ui32 rows_A,
                       const ui32 cols_A,
                       const f64 factor,
                       f64* __restrict__ matrix_A,
                       const ui32 rows_B,
                       const ui32 cols_B,
                       const f64* __restrict__ matrix_B);
#endif