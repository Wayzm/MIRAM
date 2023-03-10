#include <stdio.h>
#include <time.h>
#include "io.h"
#include "miram.h"
#include "tools.h"
#include "blas_custom.h"

#define epsilon 10e-7
int main(){

    // ENVIRONNEMENT
    const char* cmd = "export OMP_NUM_THREADS=2";
    system(cmd);

    // ARNOLDI PARAMETERS
    ui32 rows = 3; // CANNOT BE CONST BECAUSE OF BLAS
    ui32 cols = 3; // CANNOT BE CONST BECAUSE OF BLAS
    const ui32 n_krylov = 2;
	f64* __restrict__ matrix_V_C = aligned_alloc(64, sizeof(f64) * 8 * (6 + 1));
	f64* __restrict__ matrix_H_C = calloc(8 * 6, sizeof(f64));
    f64* __restrict__ matrix_V_VERIF = aligned_alloc(64, sizeof(f64) * rows * (n_krylov + 1));
	f64* __restrict__ matrix_H_VERIF = calloc(8 * 6, sizeof(f64));
	f64* __restrict__ matrix_V_M = aligned_alloc(64, sizeof(f64) * 8 * (6 + 1));
	f64* __restrict__ matrix_H_M = calloc(8 * 6, sizeof(f64));
	f64* __restrict__ matrix_A = aligned_alloc(64, sizeof(f64) * rows * cols);
	f64* __restrict__ vecteur = aligned_alloc(64, sizeof(f64) * rows);

	for(ui32 i = 0U; i < 3; ++i)
		vecteur[i] = (f64)i;

	matrix_A[0] = 1.0;
	matrix_A[1] = 1.0;
	matrix_A[2] = 1.0;
	matrix_A[3] = 4.0;
	matrix_A[4] = 2.0;
	matrix_A[5] = 1.0;
	matrix_A[6] = 9.0;
	matrix_A[7] = 3.0;
	matrix_A[8] = 1.0;

    // BLAS PARAMETERS
    ui32 INFO;
    ui32 LWORK = rows * rows; // CANNOT BE CONST BECAUSE OF BLAS
	int* __restrict__ IPIV = aligned_alloc(64, sizeof(int) * rows);
	f64* __restrict__ WORK = aligned_alloc(64, sizeof(f64) * LWORK);

    // VERIFICATION THAT RANDOMLY GENERATED MAT IS INVERSIBLE
	f64* __restrict__ matrix_AI = aligned_alloc(64, sizeof(f64) * cols * rows);
	for(ui32 i = 0U; i < rows * cols; ++i)
		matrix_AI[i] = matrix_A[i];

    // NEED MKL? COMPILES WITH IT BUT NOT WITHOUT
	dgetrf_(&rows, &cols, matrix_AI, &rows, IPIV,&INFO);
	dgetri_(&rows, matrix_AI, &cols, IPIV, WORK, &LWORK, &INFO);

	if(INFO != 0){
		perror("Matice non inversible. \n");
		exit(0);
	}

    // SOLUTION VALUES FOR THE FOLLOWING TESTS

    f64* __restrict__ Test_A = read_matrix("../bigA_test.txt", 8, 8);
    //compare_matrix(rows, cols, matrix_A, rows, cols, Test_A, epsilon);
    f64* __restrict__ Test_Q = read_matrix("../Test_Q.txt", rows, cols);
    f64* __restrict__ Test_H = read_matrix("../Test_H.txt", rows, n_krylov);

    // TESTING BLAS FUNCTIONS
    f64 result_gevv_classic = GEVV_CLASSIC(rows, 1.0, vecteur, vecteur);
    assert(result_gevv_classic == 5.0);
    f64* __restrict__ result_gemv_classic = GEMV_CLASSIC(rows, cols, 1.0,
                                                   matrix_A, rows, vecteur);
    assert(fabs(result_gemv_classic[0] - 3.0) <= epsilon);
    assert(fabs(result_gemv_classic[1] - 4.0) <= epsilon);
    assert(fabs(result_gemv_classic[2] - 5.0) <= epsilon);
    free(result_gemv_classic);

    f64 result_gemv_modified = GEMV_MODIFIED(rows, cols, 1.0, 1, matrix_A,
                                             rows, vecteur);
    assert(result_gemv_modified == 8.0);

    f64* __restrict__ result_gemm_modified = GEMM_MODIFIED(rows, cols, 1.0,
                                                           1, matrix_A, rows,
                                                           cols, matrix_A);

    assert(fabs(result_gemm_modified[0] - 6.0) <= epsilon);
    assert(fabs(result_gemm_modified[1] - 11.0) <= epsilon);
    assert(fabs(result_gemm_modified[2] - 18.0) <= epsilon);
    free(result_gemm_modified);

    // GEMM
    f64* __restrict__ gemm_A = aligned_alloc(64, sizeof(f64) * rows * cols);
    f64* __restrict__ gemm_C = aligned_alloc(64, sizeof(f64) * rows * cols);
    memcpy(gemm_A, matrix_A, rows * cols * sizeof(f64));
    GEMM_CLASSIC(rows, cols, 1.0, matrix_A, rows, cols, gemm_A, gemm_C);
    GEMM_CLASSIC_NO_C(rows, cols, 1.0, gemm_A, rows, cols, matrix_A, 0);

    // TESTING TOOLS FUNCTIONS
    f64 result_norm_vector = norm_vector(rows, vecteur);
    assert(fabs(result_norm_vector - sqrt(5)) <= epsilon);

    f64 result_norm_frobenius = norm_frobenius(rows, cols, matrix_A);
    assert(fabs(result_norm_frobenius - sqrt(115.0)) <= epsilon);

    // Testing Arnoldi Methods

	f64* __restrict__ vector = aligned_alloc(64, sizeof(f64) * 8);

	for(ui32 i = 0U; i < 8; ++i)
		vector[i] = (f64)i;
    ArnoldiProjection_Modified(8, 8, 6, vector,
                              Test_A, matrix_V_M, matrix_H_M);


    ArnoldiProjection_Classic(8, 8, 6, vector,
                               Test_A, matrix_V_C, matrix_H_C);

    // // Testing QR
    f64* __restrict__ qr_Q = aligned_alloc(64, sizeof(f64) * 6 * 6);

    f64 norm_H = norm_frobenius(8, 6, matrix_H_M);
	// #pragma omp parallel for schedule(static)
	// for(ui32 i = 0; i < 6 * 6; ++i)
	// 	matrix_H_M[i] = matrix_H_M[i]/norm_H;

    // display_matrix(matrix_H_M, 6, 6);
    for(ui32 i = 0; i < 100; ++i){
        QR_Decomposition(6, 6 - i%3, matrix_H_M, qr_Q);
        // printf("\nIter : %d \n", i);
        // f64* qr_Q_T = TRANSPOSE_MAT(6, 6, qr_Q);
        // GEMM_CLASSIC_NO_C(6, 6,1.0, qr_Q_T, 6, 6, qr_Q, 0);
        // display_matrix(qr_Q_T, 6, 6);
        // free(qr_Q_T);
    }
    // printf("\n");
    // f64* qr_Q_T = TRANSPOSE_MAT(6, 6, qr_Q);
    // GEMM_CLASSIC_NO_C(6, 6,1.0, qr_Q_T, 6, 6, qr_Q, 0);
    // display_matrix(matrix_H_M, 6, 6);
    // free(qr_Q_T);
    matrix_H_M[6 * 7 - 1] = matrix_H_M[6 * 7 - 1] / norm_H;
    f64* __restrict__ qr_vect_1 = aligned_alloc(64, sizeof(f64) * 6);
    for(ui32 i = 0; i < 6; ++i)
        qr_vect_1[i] = qr_Q[i * 6];
    f64* vect_in_A_domain = GEMV_CLASSIC(8, 6, 1.0, matrix_V_M, 6, qr_vect_1);

    f64* __restrict__ eigen_vect = GEMV_CLASSIC(8,8, 1.0, Test_A, 8, vect_in_A_domain);
    f64* __restrict__ Q_2 = aligned_alloc(64, sizeof(f64) * 36);
    memcpy(Q_2, qr_Q, sizeof(f64) * 36);

    f64* __restrict__ result = aligned_alloc(64, sizeof(f64) * 36);
    f64 tmp_q = 0;

    for(ui32 i = 0; i < 6; ++i){
        for(ui32 j = 0; j < 6; ++j){
            for(ui32 k = 0; k < 6; ++k)
                tmp_q += Q_2[k * 6 + i] * qr_Q[k * 6 + j];
            result[i * 6 + j] = tmp_q;
            tmp_q = 0;
        }
    }
    // getchar();
    // printf("\n");
    // display_matrix(matrix_V_M, 8, 7);
    // display_matrix(matrix_V_M, 8, 7);
    // printf("\n \n");
    // display_matrix(matrix_H_M, 8, 6);
    // compare_matrix(rows, cols, matrix_V_C, rows, cols, Test_Q, epsilon);
    // compare_matrix(rows, cols, matrix_V_M, rows, cols, Test_Q, epsilon);
    // compare_matrix(rows, n_krylov, matrix_H_C, rows, n_krylov, Test_H, epsilon);
    // compare_matrix(rows, n_krylov, matrix_H_M, rows, n_krylov, Test_H, epsilon);

    // END

    printf("All tests finished successfully. \n");

    free(result);
    free(Q_2);
    free(eigen_vect);
    free(vector);
    free(vect_in_A_domain);
    free(qr_vect_1);
    free(gemm_A);
    free(gemm_C);
	free(matrix_V_C);
	free(matrix_H_C);
	free(matrix_V_M);
	free(matrix_H_M);
    free(matrix_H_VERIF);
    free(matrix_V_VERIF);
    free(matrix_AI);
    free(matrix_A);
    free(Test_A);
    free(Test_Q);
    free(Test_H);
	free(vecteur);
	free(IPIV);
	free(WORK);
    // free(qr_Q);
    return 0;
}