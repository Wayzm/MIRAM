#include "miram.h"
#include "tools.h"
#include "blas_custom.h"
#include "lib.h"


void ArnoldiProjection_Modified(const ui32 rows,
								const ui32 cols,
	                   			const ui32 n_krylov,
	                   			const f64* vector,
	                   			const f64* matrix_A,
	                   			f64* restrict matrix_Q,
	                   			f64* restrict matrix_H){
	f64* __restrict__ normed_vecteur = normalization_uniform_vector(rows, vector);
	f64* __restrict__ temp = aligned_alloc(64, sizeof(f64) * rows);
	static const f64 eps = 1e-12;

	for(ui32 i = 0U; i < rows; ++i){
		matrix_Q[i * (n_krylov + 1)] = normed_vecteur[i];
	}

	for(ui32 k = 1; k < n_krylov + 1 ; ++k){

        f64* __restrict__ v = GEMM_MODIFIED(rows, n_krylov + 1, 1.0,
                                            k - 1, matrix_Q, rows, cols,
                                            matrix_A);
		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i){
				temp[i] = matrix_Q[i * (n_krylov + 1) + j];
			}
			matrix_H[j * n_krylov + k - 1] = GEVV_CLASSIC(rows, 1.0, temp, v); 

			for(ui32 i = 0U; i < rows; ++i){
				v[i] = v[i] - matrix_H[j * n_krylov + k - 1] * temp[i];
			}
		}

		matrix_H[k * n_krylov + k - 1] = norm_vector(rows, v);

		if(matrix_H[k * n_krylov + k - 1] > eps){
			for(ui32 j = 0U; j < rows; ++j){
				matrix_Q[j * (n_krylov + 1) + k] = v[j] / matrix_H[k * n_krylov + k - 1];
			}
		}
		else{
			printf(" Perfect solution !\n");
			break;
		}       
        free(v);
	}
	free(normed_vecteur);
	free(temp);
}

void ArnoldiProjection_Classic(const ui32 rows,
								 const ui32 cols,
	                   			 const ui32 n_krylov,
	                   			 const f64* vector,
	                   			 const f64* matrix_A,
	                   			 f64* restrict matrix_Q,
	                   			 f64* restrict matrix_H){
	f64* restrict normed_vecteur = normalization_uniform_vector(rows, vector);
	f64* restrict temp = aligned_alloc(64, sizeof(f64) * rows);
	static const f64 eps = 1e-12;

	for(ui32 i = 0U; i < rows; ++i){
		matrix_Q[i * (n_krylov + 1)] = normed_vecteur[i];
	}

	for(ui32 k = 1; k < n_krylov + 1; ++k){

        f64* __restrict__ v = GEMM_MODIFIED(rows, n_krylov + 1, 1.0,
                                            k - 1, matrix_Q, rows, cols,
                                            matrix_A);
		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i)
				temp[i] = matrix_Q[i * (n_krylov + 1) + j];
			matrix_H[j * n_krylov + k - 1] = GEVV_CLASSIC(rows, 1.0, temp, v);
		}

		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i)
				v[i] = v[i] - matrix_H[j * n_krylov + k - 1] * matrix_Q[i * (n_krylov + 1) + j];
		}

		matrix_H[k * n_krylov + k - 1] = norm_vector(rows,v);

		if(matrix_H[k * n_krylov + k - 1] > eps){
			for(ui32 j = 0U; j < rows; ++j){
				matrix_Q[j * (n_krylov + 1) + k] = v[j] / matrix_H[k * n_krylov + k - 1];
			}
		}
		else{
			printf("Perfect solution !");
			break;
		}
        free(v);
	}
	free(normed_vecteur);
	free(temp);
}