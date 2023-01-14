#ifndef BLAS_CuSTOM_H
#define BLAS_CUSTOM_H

#include "lib.h"

/***
 * GEVV_CLASSIC computes a normal vector * vector : a = b * x * y
 * GEMV_CLASSIC computes a normal matrix * vector : z = a * M * x 
 * GEMV_MODIFIED computes the nth colums of the matrix * vector : a = b * M[i + cols] * x
 * 
 ***/
double GEVV_CLASSIC(const unsigned int size,
                      const double factor,
				      const double* vecteur_x,
				      const double* vecteur_y);

double* GEMV_CLASSIC(const unsigned int rows,
                      const unsigned int cols,
                      const double factor,
                      const double* __restrict__ matrix,
                      const unsigned int size_vec,
                      const double* __restrict__ vector);

double* GEMV_MODIFIED(const unsigned int rows,
                     const unsigned int cols,
                     const double factor,
                     const double* __restrict__ matrix,
                     const double* __restrict__ vector);


#endif