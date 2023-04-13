#ifndef TOOLS_H
#define TOOLS_H

#include "lib.h"

/***
 * gen_matrix will generates a random matrix of size rows * cols
 * d_rand will generates a random f64 value
 * norm_vector will return the norm 2 of the vector
 * norm_frobenius will return the norm 2 of the matrix
 * normalization_uniform_vector will return an uniformally normed vector
 * compare_matrix will verify if each value is the same within an error margin
 ***/
f64* gen_matrix(const ui32 rows,
				  const ui32 cols);
f64 d_rand();

f64 norm_vector(const ui32 size,
					 const f64* vecteur);

f64 norm_frobenius(const ui32 rows,
					   const ui32 cols,
					   const f64* matrix);


f64* normalization_uniform_vector(const ui32 size,
								     const f64* vecteur);

void compare_matrix(const unsigned rows_A,
				   const unsigned cols_A,
				   const f64* __restrict__ matrix_A,
                   const unsigned rows_B,
                   const unsigned cols_B,
				   const f64* __restrict__ matrix_B,
				   const f64 eps);

f64 verify_matrix(const ui32 rows,
					const ui32 cols,
					const f64* __restrict__ matrix);

#endif