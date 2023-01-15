#include "tools.h"
#include "lib.h"

inline f64 d_rand(){
	return (f64)rand() * 100;
}

f64* gen_atrix(const ui32 rows,
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
	f64 norme = 0;
	norme = DotProduct(size, vecteur, vecteur);
	norme = sqrt(norme);
	return norme;
}

f64 norm_robenius(const ui32 rows,
				  const ui32 cols,
				  const f64* matrix){

	assert(rows != 0 && cols != 0);
	f64 norme = 0;
	for(ui32 i = 0U; i < rows * cols; ++i)
		norme += matrix[i] * matrix[i];
	norme = sqrt(norme);
	return norme;
}

f64* normalization_uniform_vector(const ui32 size,
								  const f64* vecteur){
	const f64 max = Norme_Vecteur(size, vecteur);
	f64* restrict normed_vector = aligned_alloc(64, sizeof(f64) * size); 
	for(ui32 i = 0U; i < size; ++i)
		normed_vector[i] = vecteur[i] / max;
	return normed_vector;
}

void compare_matrix(const ui32 rows_A,
				    const ui32 cols_A,
				    const f64* restrict matrix_A,
                    const ui32 rows_B,
                    const ui32 cols_B,
				    const f64* restrict matrix_B,
					const f64 eps){

	assert(rows_A != 0 && cols_A != 0);
    assert(rows_A == rows_B);
    assert(cols_A == cols_B);

	for(ui32 i = 0U; i < rows_A; ++i){
		for(ui32 j = 0U; j < cols_A; ++j){
			const f64 comp = abs(matrix_A[i * cols_A] - matrix_B[i * cols_B + j]);
			assert(comp <= eps);
		}
	}
}

f64 verify_matrix(const ui32 rows,
				  const ui32 cols,
				  const f64* restrict matrix_A){

	assert(rows != 0 && cols != 0);
	printf("Norme de Frobenius de la matrice : %lf \n", norme_frobenius(rows, cols, matrix_A));
	f64* restrict matrix = aligned_alloc(64, sizeof(f64) * rows * cols);
	f64 matmat;

	for(ui32 i = 0U; i < rows; ++i){
		for(ui32 j = 0U; j < cols;  ++j){
			matmat = 0;
			for(ui32 k = 0U; k < rows; ++k){
				matmat += matrix_A[k * cols + j] * matrix_A[k * cols + j];
			}
			matrix[i * cols + j] = matmat;
		}
	}
	f64 norme = norme_frobenius(rows, cols, matrix);
	
	free(matrix);
	return norme;
}