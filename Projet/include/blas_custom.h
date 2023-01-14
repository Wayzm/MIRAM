#ifndef BLAS_CuSTOM_H
#define BLAS_CUSTOM_H

#include "lib.h"

/***
 * GEVV_CLASSIC computes a normal vector * vector : a = b * x * y
 * GEMV_CLASSIC computes a normal matrix * vector : z = a * M * x 
 * GEMV_MODIFIED computes the nth colums of the matrix * vector : a = b * M[i + cols] * x
 * 
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
                     const f64* __restrict__ matrix,
                     const ui32 size_vec,
                     const f64* __restrict__ vector);


#endif