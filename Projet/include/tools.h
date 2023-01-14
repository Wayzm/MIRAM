#ifndef TOOLS_H
#define TOOLS_H

#include "lib.h"

/***
 * gen_matrix will generates a random matrix of size rows * cols
 * d_rand will generates a random double value
 * norm_vector will return the norm 2 of the vector
 * norm_frobenius will return the norm 2 of the matrix
 * normalization_uniform_vector will return an uniformally normed vector
 ***/
double* gen_matrix(const unsigned int rows,
				  const unsigned int cols);
double d_rand();

double norm_vector(const unsigned int size,
					 const double* vecteur);

double norm_frobenius(const unsigned int rows,
					   const unsigned int cols,
					   const double* matrix);

double* normalization_uniform_vector(const unsigned int size,
								     const double* vecteur);

void compare_matrix(const unsigned rows_A,
				   const unsigned cols_A,
				   const double* __restrict__ matrix_A,
                   const unsigned rows_B,
                   const unsigned cols_B,
				   const double* __restrict__ matrix_B);

double verify_matrix(const unsigned int rows, 
					const unsigned int cols,
					const double* __restrict__ matrix);

#endif