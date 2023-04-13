#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <float.h>
#include "io.h"
#include "miram.h"
#include "tools.h"
#include "blas_custom.h"

#define BILLION 10e9
#define eps 10e-4

int main(int argc, char** argv){

	/*** VARIABLES ***/
	ui32 m = 0;
	ui32 cols = 0;
	ui32 rows = 0;
	ui32 n;
	f64 tmp = 0;
	f64 f_m;
	ui32 status_iram = 0;
	int rank, comm_size, best_rank, res;
	f64* compare_array;
	f64 error = DBL_MAX;

	if(argc < 5){
		perror("<exec> <file path> <rows> <cols> <nbr of eigen values>");
		exit(0);
	}
	/*** SET UP ***/
	if(access(argv[1], R_OK ) != 0 || access(argv[1],F_OK) != 0)
    {
      // file doesn't exist or can't be read
      perror("File was not found or does not have read permission!\nPlease, try again.\n");
      exit(0);
    }

	rows = atoi(argv[2]);
	cols = atoi(argv[3]);
	m = atoi(argv[4]);
	n = 2 * m; // KRYLOV SUBDOMAIN RANK
	const ui32 shift = n - m; // Number of shift in QR decomposition

	/*** MATRIXES ***/
	f64* __restrict__ matrix_A = read_matrix(argv[1], rows, cols);
	f64* __restrict__ matrix_V = aligned_alloc(64, sizeof(f64) * rows * (n + 1));
	f64* __restrict__ matrix_H = aligned_alloc(64, sizeof(f64) * rows * n);
	f64* __restrict__ Final_eigen_vectors = aligned_alloc(64, sizeof(f64) * rows * n);
	f64* __restrict__ matrix_Q = aligned_alloc(64, sizeof(f64) * n * n);

	MPI_Init(&argc, &argv);

	// MPI SET UP
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	srand(time(NULL) + rank);
	f64* __restrict__ init_vector = aligned_alloc(64, sizeof(f64) * rows);

	// /*** IRAM ***/
	for(ui32 i = 0; i < rows; ++i)
		init_vector[i] = (f64)(rand()%1000);

	while(status_iram == 0){
		ArnoldiProjection_Modified(rows, cols, n, init_vector, matrix_A,
									matrix_V, matrix_H);

		// last value of matrix_H
		f_m = matrix_H[n * n + n - 1];

		MPI_Barrier(MPI_COMM_WORLD);
		// Get all the f_m from the differents process
		if(rank == 0){
			compare_array = malloc(sizeof(f64) * comm_size);
			compare_array[0] = f_m;
		}

		res = MPI_Gather(&f_m, 1, MPI_DOUBLE, &compare_array[rank], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (res != MPI_SUCCESS){
			perror("MPI_Comm_rank failedfor Gather of f_m \n");
			exit (0);
		}
		// Check which one has the lowest value
		if(rank == 0){
			best_rank = 0;
			f64 lowest_fm = compare_array[0];
			for(ui32 i = 1; i < comm_size; ++i){
				if(compare_array[i] <= lowest_fm){
					lowest_fm = compare_array[i];
					best_rank = i;
				}
			}
		}
		res = MPI_Bcast(&best_rank, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (res != MPI_SUCCESS){
			perror("MPI_Comm_rank failed for Bcast of best rank\n");
			exit (0);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		res = MPI_Bcast(&matrix_H[0], rows * n, MPI_DOUBLE, best_rank, MPI_COMM_WORLD);
		if (res != MPI_SUCCESS){
			perror("MPI_Comm_rank failed for Bcast of H\n");
			exit (0);
		}

		res = MPI_Bcast(&matrix_V[0], rows * (n + 1), MPI_DOUBLE, best_rank, MPI_COMM_WORLD);
		if (res != MPI_SUCCESS){
			perror("MPI_Comm_rank failed for Bcast of V\n");
			exit (0);
		}

		// Remove the last column of V
		f64* __restrict__ V_krylov = matrix_col_modification(rows, n + 1, matrix_V, n);
		// Matrix q for restart vector
		f64* __restrict__ q = calloc(n, sizeof(f64));
		q[n - 1] = 1.0;

		// QR Decomposition of H to get the eigenvalues and eigen vectors
		for(ui32 i = 1; i <= 50; ++i){
			QR_Decomposition(n, n - i%shift, matrix_H, matrix_Q);
			// matrix Q stores the eigen vectors at each step so we multiply with the transition matrix V
			GEMM_CLASSIC_NO_C(rows, n, 1.0, V_krylov, n, n, matrix_Q, 1);
			GEMV_CLASSIC_NO_R(n, n, matrix_Q, n, q, 1);
		}

		/*** Checking  with Ritz coefficient ***/
		// We take the first eigen vector in the A's domain
		f64* __restrict__ Eigen_Vector = aligned_alloc(64, sizeof(f64) * rows);
		for(ui32 i = 0; i < rows; ++i)
			Eigen_Vector[i] = V_krylov[i * n];
		f64* __restrict__ EV_bis = aligned_alloc(64, sizeof(f64) * rows);
		memcpy(EV_bis, Eigen_Vector, sizeof(f64) * rows);
		// Now we compute the ritz values
		GEMV_CLASSIC_NO_R(rows, cols, matrix_A, rows, Eigen_Vector, 0);
		const f64 dominant_value = matrix_H[0]; // 1st eigen value
		for(ui32 i = 0; i < rows; ++i){
			EV_bis[i] *= dominant_value;
		}
		const f64 ritz_coef = fabs(EV_bis[0] - Eigen_Vector[0])/fabs(EV_bis[0]);
		free(Eigen_Vector);
		free(EV_bis);

		if(ritz_coef < eps){
			memcpy(Final_eigen_vectors, V_krylov, sizeof(f64) * rows * n);
			MPI_Bcast(&Final_eigen_vectors[0], rows * n, MPI_DOUBLE, rank, MPI_COMM_WORLD);
			free(V_krylov);
			free(q);
			if(rank == 0){
				printf("Dominant eigen value : %lf\n", dominant_value);
				printf("Associated eigen vector : \n");
				for(ui32 i = 0; i < rows; ++i){
					printf("%lf\n", Final_eigen_vectors[i]);
				}
			}
			status_iram = 1;

		}
		/*** Checking with Ritz Coefficient ***/

		/*** Restart vector ***/
		f64* __restrict__ restart_vector = aligned_alloc(64, sizeof(f64) * rows);
		const f64 q_factor = matrix_Q[(n - 1) * n + rank%n];
		#pragma omp parallel for schedule(static)
		for(ui32 i = 0; i < rows; ++i){
			tmp = 0;
			for(ui32 j = 0; j < n; ++j){
				tmp += V_krylov[i * n + j] * q[j] + matrix_V[i * (n + 1) + rank%n];
			}
			restart_vector[i] = f_m * tmp;
		}
		memcpy(init_vector, restart_vector, sizeof(f64) * rows);
		free(restart_vector);
		/***Restart vector ***/
	}
	free(init_vector);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	/*** FREE STRUCTURES ***/
	free(matrix_A);
	free(matrix_V);
	free(matrix_H);
	free(Final_eigen_vectors);
	free(matrix_Q);
	/*** FREE STRUCTURES ***/
	return 0;
}