#ifndef MIRAM_H
#define MIRAM_H

#include "lib.h"
/***
 * ArnoldiProjection_Modified computes the reduced model with GEVV and GEMV_Modified in the same loop
 * ArnoldiProjection_Classic computes the reduced model with GEVV and GEMV_Modified in different loops
*/
void ArnoldiProjection_Modified(const ui32 rows_A,
								const ui32 cols_A,
								const ui32 n_krylov,
								const f64* vecteur,
								const f64* __restrict__ matrix_A,
								f64* __restrict__ matrix_V,
								f64* __restrict__ matrix_H);

void ArnoldiProjection_Classic(const ui32 rows_A,
								 const ui32 cols_A,
								 const ui32 n_krylov,
								 const f64* vecteur,
								 const f64* __restrict__ matrix_A,
								 f64* __restrict__ matrix_V,
								 f64* __restrict__ matrix_H);

void QR_Algorithm(const ui32 rows,
				  const ui32 n_krylov,
				  f64* __restrict__ matrix_H,
				  f64* __restrict__ matrix_V);

void IRAM(const ui32 rows_A,
		  const ui32 cols_A,
		  const ui32 n_krylov,
		  const ui32 shift,
		  const f64* __restrict__ Init_vector,
		  const f64* __restrict__ matrix_A,
		  f64* __restrict__ matrix_V,
		  f64* __restrict__ matrix_H,
		  const ui32 nbr_eigenvalues,
		  f64* __restrict__ eigenvalues);

#endif