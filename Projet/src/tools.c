#include "tools.h"
#include "blas_custom.h"
#include "lib.h"

inline f64 d_rand(){
	return (f64)rand() * 100;
}

f64* gen_matrix(const ui32 rows,
				const ui32 cols){

	assert(rows != 0 && cols != 0);

	f64* restrict matrix = aligned_alloc(64, sizeof(f64) * cols * rows);

	if(rows != 1){
		for(ui32 i = 0U; i < rows; ++i){
			for(ui32 j = 0U; j < cols; ++j){
				if(i == j)
					matrix[i * cols + j] = d_rand() + 1;
				else
					matrix[i * cols +j] = d_rand();
			}
		}
	}
	else{
		for(ui32 i = 0U; i < cols; ++i)
			matrix[i] = d_rand() * 100;
	}

	return matrix;

}

f64 norm_vector(const ui32 size,
				const f64* vecteur){
	assert(size != 0);
	f64 norme = 0;
	for(ui32 i = 0; i < size; ++i)
		norme += vecteur[i] * vecteur[i];
	norme = sqrt(norme);
	return norme;
}

f64 norm_frobenius(const ui32 rows,
				   const ui32 cols,
				   const f64* matrix){

	assert(rows != 0 && cols != 0);
	f64 norme = 0;
	for(ui32 i = 0; i < rows * cols; ++i)
		norme += matrix[i] * matrix[i];
	norme = sqrt(norme);
	return norme;
}

f64* normalization_uniform_vector(const ui32 size,
								  const f64* vecteur){
	assert(size != 0);
	const f64 max = norm_vector(size, vecteur);
	f64* restrict normed_vector = aligned_alloc(64, sizeof(f64) * size);
	#pragma omp parallel for schedule(dynamic, 1)
	for(ui32 i = 0; i < size; ++i)
		normed_vector[i] = vecteur[i] / max;
	return normed_vector;
}

void compare_matrix(const ui32 rows_A,
				    const ui32 cols_A,
				    const f64* __restrict__ matrix_A,
                    const ui32 rows_B,
                    const ui32 cols_B,
				    const f64* __restrict__ matrix_B,
					const f64 eps){

	assert(rows_A != 0 && cols_A != 0);
    assert(rows_A == rows_B);
    assert(cols_A == cols_B);

	#pragma omp parallel for schedule(dynamic, 1)
	for(ui32 i = 0; i < rows_A; ++i){
		for(ui32 j = 0; j < cols_A; ++j){
			const f64 comp = fabs(matrix_A[i * cols_A + j] - matrix_B[i * cols_B + j]);
			assert(comp <= eps);
		}
	}
}

f64 verify_matrix(const ui32 rows,
				  const ui32 cols,
				  const f64* __restrict__ matrix_A){

	assert(rows != 0 && cols != 0);
	printf("Norme de Frobenius de la matrice : %lf \n", norm_frobenius(rows, cols, matrix_A));
	f64* restrict matrix = aligned_alloc(64, sizeof(f64) * rows * cols);
	f64 matmat;

	#pragma omp parallel for schedule(dynamic, 1) private(matmat)
	for(ui32 i = 0; i < rows; ++i){
		for(ui32 j = 0; j < cols;  ++j){
			matmat = 0;
			for(ui32 k = 0; k < rows; ++k){
				matmat += matrix_A[k * cols + j] * matrix_A[k * cols + j];
			}
			matrix[i * cols + j] = matmat;
		}
	}
	f64 norme = norm_frobenius(rows, cols, matrix);

	free(matrix);
	return norme;
}

f64* matrix_col_modification(const ui32 rows,
							 const ui32 cols,
						 	 const f64* __restrict__ matrix,
						 	 const ui32 line){

	f64* __restrict__ tronc_mat = aligned_alloc(64, sizeof(f64) * rows * (cols - 1));
	for(ui32 i = 0; i < rows; ++i){
		for(ui32 j = 0; j < cols; ++j){
			if(j == line)
				continue;
			tronc_mat[i * (cols - 1) + j] = matrix[i * cols + j];
		}
	}
	return tronc_mat;
}