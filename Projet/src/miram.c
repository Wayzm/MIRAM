#include "miram.h"
#include "tools.h"
#include "io.h"
#include "blas_custom.h"
#include "lib.h"


void ArnoldiProjection_Modified(const ui32 rows,
								const ui32 cols,
	                   			const ui32 n_krylov,
	                   			const f64* vector,
	                   			const f64* matrix_A,
	                   			f64* restrict matrix_V,
	                   			f64* restrict matrix_H){
	f64* __restrict__ normed_vecteur = normalization_uniform_vector(rows, vector);
	f64* __restrict__ temp = aligned_alloc(64, sizeof(f64) * rows);
	static const f64 eps = 1e-12;

	assert(n_krylov < cols);

	for(ui32 i = 0U; i < rows; ++i){
		matrix_V[i * (n_krylov + 1)] = normed_vecteur[i];
	}

	for(ui32 k = 1; k < n_krylov + 1 ; ++k){

        f64* __restrict__ v = GEMM_MODIFIED(rows, n_krylov + 1, 1.0,
                                            k - 1, matrix_V, rows, cols,
                                            matrix_A);

		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i){
				temp[i] = matrix_V[i * (n_krylov + 1) + j];
			}
			matrix_H[j * n_krylov + k - 1] = GEVV_CLASSIC(rows, 1.0, temp, v);

			for(ui32 i = 0U; i < rows; ++i){
				v[i] = v[i] - matrix_H[j * n_krylov + k - 1] * temp[i];
			}
		}

		matrix_H[k * n_krylov + k - 1] = norm_vector(rows, v);
		if(matrix_H[k * n_krylov + k - 1] > eps){
			for(ui32 j = 0U; j < rows; ++j){
				matrix_V[j * (n_krylov + 1) + k] = v[j] / matrix_H[k * n_krylov + k - 1];
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
	                   			 f64* restrict matrix_V,
	                   			 f64* restrict matrix_H){
	f64* restrict normed_vecteur = normalization_uniform_vector(rows, vector);
	f64* restrict temp = aligned_alloc(64, sizeof(f64) * rows);
	static const f64 eps = 1e-12;

	assert(n_krylov < cols);

	for(ui32 i = 0U; i < rows; ++i){
		matrix_V[i * (n_krylov + 1)] = normed_vecteur[i];
	}

	for(ui32 k = 1; k < n_krylov + 1; ++k){

        f64* __restrict__ v = GEMM_MODIFIED(rows, n_krylov + 1, 1.0,
                                            k - 1, matrix_V, rows, cols,
                                            matrix_A);
		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i)
				temp[i] = matrix_V[i * (n_krylov + 1) + j];
			matrix_H[j * n_krylov + k - 1] = GEVV_CLASSIC(rows, 1.0, temp, v);
		}

		for(ui32 j = 0U; j < k; ++j){
			for(ui32 i = 0U; i < rows; ++i)
				v[i] = v[i] - matrix_H[j * n_krylov + k - 1] * matrix_V[i * (n_krylov + 1) + j];
		}

		matrix_H[k * n_krylov + k - 1] = norm_vector(rows,v);

		if(matrix_H[k * n_krylov + k - 1] > eps){
			for(ui32 j = 0U; j < rows; ++j){
				matrix_V[j * (n_krylov + 1) + k] = v[j] / matrix_H[k * n_krylov + k - 1];
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

// Strat, on a une variable qui sera modifié si le coef de ritz est bas
// une boucle while qui itère tant que la précision n'est pas celle demandée
// une boucle while imbriqué dont on sort pour MPI_Barrier et faire les coms
void QR_Algorithm(const ui32 rows,
				  const ui32 n_krylov,
				  f64* __restrict__ matrix_H,
				  f64* __restrict__ matrix_V){
	// No shift
	ui32 count = 0;

	// Normalisation of H to avoid overflows
	f64 norm_H = norm_frobenius(n_krylov, n_krylov, matrix_H);
	#pragma omp parallel for schedule(static)
	for(ui32 i = 0; i < n_krylov * n_krylov; ++i)
		matrix_H[i] = matrix_H[i]/norm_H;

	// Final matrix Qf for QR decomposition
	f64* __restrict__ matrix_Qf = calloc(n_krylov * n_krylov, sizeof(f64));
	#pragma omp parallel for schedule(static)
	for(ui32 j = 0; j < n_krylov * n_krylov; ++j){
		if(count%n_krylov + 1 == 0)
			matrix_Qf[j] = 1.0;
		count ++;
	}

	count = 0;

	f64* __restrict__ matrix_R = aligned_alloc(64, sizeof(f64) * n_krylov * n_krylov);

	// Computing Qf for QR decomposition
	for(ui32 i = 0; i < n_krylov - 1; ++i){
		f64* __restrict__ matrix_Q = calloc(n_krylov * n_krylov, sizeof(f64));
		for(ui32 j = 0; j < n_krylov; ++j){
			for(ui32 k = 0; k < n_krylov; ++k){
				if (j == k)
					matrix_Q[j * n_krylov + k] = 1.0;
			}
		}
		// We select the small square needed for the QR decomposition
		const f64 h_11 = matrix_H[i * n_krylov + i];
		const f64 h_21 = matrix_H[i * n_krylov + n_krylov + i];
		const f64 gamma = h_11 / (sqrt(h_21 * h_21 + h_11 * h_11));
		const f64 sigma = h_21 / (sqrt(h_21 * h_21 + h_11 * h_11));
		matrix_Q[i * n_krylov + i] = gamma;
		matrix_Q[i * n_krylov + n_krylov + i + 1] = gamma;
		matrix_Q[i * n_krylov + i + 1] = sigma;
		matrix_Q[i * n_krylov + n_krylov + i] = -sigma;


		GEMM_CLASSIC(n_krylov, n_krylov, 1.0, matrix_Q, n_krylov, n_krylov, matrix_H, matrix_R);
		memcpy(matrix_H, matrix_R, n_krylov * n_krylov * sizeof(f64));
		GEMM_CLASSIC_NO_C(n_krylov, n_krylov, 1.0, matrix_Qf, n_krylov, n_krylov, matrix_Q);
		free(matrix_Q);
	}
	free(matrix_R);
	free(matrix_Qf);
}

void IRAM(const ui32 rows_A,
		  const ui32 cols_A,
		  const ui32 n_krylov,
		  const ui32 shift,
		  const f64* __restrict__ Init_vector,
		  const f64* __restrict__ matrix_A,
		  f64* __restrict__ matrix_V,
		  f64* __restrict__ matrix_H,
		  const ui32 nbr_eigenvalues,
		  f64* __restrict__ eigenvalues){

	ArnoldiProjection_Modified(rows_A, cols_A, n_krylov, Init_vector, matrix_A,
								matrix_V, matrix_H);

	assert(nbr_eigenvalues <= n_krylov);
	f64* restrict temp = aligned_alloc(64, sizeof(f64) * nbr_eigenvalues);

	//
}