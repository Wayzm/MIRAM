#include <stdio.h>
#include <time.h>
#include "io.h"
#include "miram.h"
#include "tools.h"
#include "blas_custom.h"

#define BILLION 10e9

int main(int argc, char** argv){

	/*** VARIABLES ***/
	char matrix_path[256];
	ui32 number_of_eigenvalues = 0;
	ui32 cols = 0;
	ui32 rows = 0;
	ui32 n_krylov;

	/*** SET UP ***/
	printf("Input path to the file with the matrix : \n");
	scanf("%s", matrix_path);
	if(access(matrix_path, R_OK ) != 0 || access(matrix_path,F_OK) != 0)
    {
      // file doesn't exist or can't be read
      perror("File was not found or does not have read permission!\nPlease, try again.\n");
      exit(0);
    }

	printf("Input the number of rows and cols as such: <unsigned int> <unsigned int> \n");
	scanf("%d %d", &rows, &cols);
	if(rows == 0 || cols == 0){
		perror("Rows and Cols cannot be null. \n");
		exit(0);
	}
	printf("Input the number of eigen values/eigen vectors needed : \n");
	scanf("%d", &number_of_eigenvalues);
	if(number_of_eigenvalues == 0){
		perror("The number of eigenvalues cannot be null.\n");
		exit(0);
	}
	n_krylov = 2 * number_of_eigenvalues; // KRYLOV SUBDOMAIN RANK
	const ui32 shift = n_krylov - number_of_eigenvalues; // Number of shift in QR decomposition
	printf("Shift : %d \n", shift);
	/*** MATRIXES ***/
	f64* __restrict__ matrix_A = read_matrix(matrix_path, rows, cols);
	f64* __restrict__ matrix_V = aligned_alloc(64, sizeof(f64) * rows * (n_krylov + 1));
	f64* __restrict__ matrix_H = aligned_alloc(64, sizeof(f64) * rows * n_krylov);
	f64* __restrict__ init_vector = aligned_alloc(64, sizeof(f64) * rows);

	// /*** IRAM ***/
	for(ui32 i = 0; i < rows; ++i)
		init_vector[i] = (f64)i;

	ArnoldiProjection_Modified(rows, cols, n_krylov, init_vector, matrix_A,
								matrix_V, matrix_H);

	// Remove the last column of V
	f64* __restrict__ V_krylov = matrix_col_modification(rows, n_krylov + 1, matrix_V, n_krylov);
	// Matrix Q for QR
	f64* __restrict__ matrix_Q = aligned_alloc(64, sizeof(f64) * n_krylov * n_krylov);
	for(ui32 i = 0; i < shift; ++i){
		QR_Decomposition(n_krylov, i, matrix_H, matrix_Q);
		GEMM_CLASSIC_NO_C(rows, n_krylov, 1.0, V_krylov, n_krylov, n_krylov, matrix_Q, 1);
	}

	/*** FREE STRUCTURES ***/
	free(matrix_A);
	free(matrix_V);
	free(matrix_H);
	free(init_vector);
	free(V_krylov);
	free(matrix_Q);
	/*** FREE STRUCTURES ***/
	return 0;
}