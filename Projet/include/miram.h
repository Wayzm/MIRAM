#ifndef MIRAM_H
#define MIRAM_H

#include "lib.h"
/***
 * ArnoldiProjection_Modified computes the reduced model with GEVV and GEMV_Modified in the same loop
 * ArnoldiProjection_Classic computes the reduced model with GEVV and GEMV_Modified in different loops
*/
void ArnoldiProjection_Modified(const unsigned int rows_A,
								const unsigned int cols_A,
								const unsigned int n_krylov,
								const double* vecteur,
								const double* __restrict__ matrix_A,
								double* __restrict__ matrix_Q,
								double* __restrict__ matrix_H);

void ArnoldiProjection_Classic(const unsigned int rows_A,
								 const unsigned int cols_A,
								 const unsigned int n_krylov,
								 const double* vecteur,
								 const double* __restrict__ matrix_A,
								 double* __restrict__ matrix_V,
								 double* __restrict__ matrix_H);


#endif