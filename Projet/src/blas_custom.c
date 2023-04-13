#include "blas_custom.h"

// vector_x * vector_y
f64 GEVV_CLASSIC(const ui32 size,
                 const f64 factor,
				 const f64* vector_x,
				 const f64* vector_y){

    f64 result = 0;

    for(ui32 i = 0; i < size; ++i){
        result += factor * vector_x[i] * vector_y[i];
    }

    return result;
}

// factor * M * vector
f64* GEMV_CLASSIC(const ui32 rows,
                  const ui32 cols,
                  const f64 factor,
                  const f64* __restrict__ matrix,
                  const ui32 size_vec,
                  const f64* __restrict__ vector){

    assert(cols == size_vec);
    f64* __restrict__ result = aligned_alloc(64, sizeof(f64) * rows);
    f64 tmp;
    #pragma omp parallel for schedule(dynamic, 1) private(tmp)
    for(ui32 i = 0; i < rows; ++i){
        tmp = 0;
        for(ui32 j = 0; j < size_vec; ++j)
            tmp += factor * matrix[i * cols + j] * vector[j];
        result[i] = tmp;
    }
    return result;
}

void GEMV_CLASSIC_NO_R(const ui32 rows,
                       const ui32 cols,
                       const f64* __restrict__ matrix,
                       const ui32 size_vec,
                       f64* __restrict__ vector,
                       const ui32 mode){
    assert(cols == size_vec && rows == size_vec);
    if(mode == 0){ // A*v
        f64* __restrict__ tmp = malloc(sizeof(f64) * size_vec);
        f64 value = 0;
        #pragma omp parallel for schedule(static) private(value)
        for(ui32 i = 0; i < rows; ++i){
            for(ui32 j = 0; j < cols; ++j){
                value += vector[j] * matrix[i * cols + j];
            }
            tmp[i] = value;
            value = 0;
        }
        memcpy(vector, tmp, sizeof(f64) * size_vec);
        free(tmp);
    }
    else { // v * A
        f64* __restrict__ tmp = malloc(sizeof(f64) * size_vec);
        f64 value = 0;
        #pragma omp parallel for schedule(static) private(value)
        for(ui32 i = 0; i < cols; ++i){
            for(ui32 j = 0; j < rows; ++j){
                value += vector[j] * matrix[j * cols + i];
            }
            tmp[i] = value;
            value = 0;
        }
        memcpy(vector, tmp, sizeof(f64) * size_vec);
        free(tmp);
    }
}

// factor * M[i;0...N] * vector
f64 GEMV_MODIFIED(const ui32 rows,
                  const ui32 cols,
                  const f64 factor,
                  const ui32 increment,
                  const f64* __restrict__ matrix,
                  const ui32 size_vec,
                  const f64* __restrict__ vector){
    assert(rows == size_vec);
    f64 result = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:result)
    for(ui32 i = 0; i < size_vec; ++i)
        result += factor * matrix[i * cols + increment] * vector[i];

    return result;
}

// factor * A[0....M:increment] * B
f64* GEMM_MODIFIED(const ui32 rows_A,
                   const ui32 cols_A,
                   const f64 factor,
                   const ui32 increment_A,
                   const f64* __restrict__ matrix_A,
                   const ui32 rows_B,
                   const ui32 cols_B,
                   const f64* __restrict__ matrix_B){

    assert(rows_A != 0 && cols_A != 0);
    assert(rows_A == rows_B);
    f64* __restrict__ result = aligned_alloc(64, sizeof(f64) * rows_A);

    #pragma omp parallel for schedule(dynamic, 1)
	for(ui32 i = 0U; i < rows_A; ++i){
        result[i] = 0;
		for(ui32 j = 0U; j < cols_B; ++j){
			result[i] += matrix_A[j * cols_A + increment_A] * matrix_B[i * cols_B + j];
		}
	}
    return result;
}

void GEMM_CLASSIC(const ui32 rows_A,
                  const ui32 cols_A,
                  const f64 factor,
                  const f64* __restrict__ matrix_A,
                  const ui32 rows_B,
                  const ui32 cols_B,
                  const f64* __restrict__ matrix_B,
                  f64* __restrict__ matrix_C){
    assert(cols_A == rows_B);
    f64 tmp = 0;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1) private(tmp)
        for(ui32 i = 0; i < rows_A; ++i){
            for(ui32 j = 0; j < cols_B; ++j){
                for(ui32 k = 0; k < cols_A; ++k){
                    tmp += matrix_A[i * cols_A + k] * matrix_B[j + cols_B * k];
                }
                matrix_C[i * cols_B + j] = factor * tmp;
                tmp = 0;
            }
        }
    }
}

// / ! \ This function is tailor made to compute the Givens rotation like so : G5 * G4 * G3 * G2 * G1
void GEMM_CLASSIC_NO_C(const ui32 rows_A,
                       const ui32 cols_A,
                       const f64 factor,
                       f64* __restrict__ matrix_A,
                       const ui32 rows_B,
                       const ui32 cols_B,
                       const f64* __restrict__ matrix_B,
                       const ui32 version){
    f64 tmp = 0;
    f64* __restrict__ matrix_C = malloc(sizeof(f64) * rows_A * cols_A);
    // The modified matrix is on the right side
    if(version == 0){
        assert(cols_B == rows_A && cols_B == cols_A);
        for(ui32 i = 0; i < rows_B; ++i){
            for(ui32 j = 0; j < cols_A; ++j){
                for(ui32 k = 0; k < cols_B; ++k){
                    tmp += matrix_B[i * cols_B + k] * matrix_A[k + cols_A * j];
                }
                matrix_C[i * cols_B + j] = factor * tmp;
                tmp = 0;
            }
        }
    }
    else{ // left side
        assert(cols_A == rows_B && cols_A == cols_B);
        for(ui32 i = 0; i < rows_A; ++i){
            for(ui32 j = 0; j < cols_B; ++j){
                for(ui32 k = 0; k < cols_A; ++k){
                    tmp += matrix_A[i * cols_A + k] * matrix_B[k + cols_B * j];
                }
                matrix_C[i * cols_B + j] = factor * tmp;
                tmp = 0;
            }
        }
    }
    memcpy(matrix_A, matrix_C, rows_A * cols_A * sizeof(f64));
    free(matrix_C);
}

f64* TRANSPOSE_MAT(const ui32 rows,
                   const ui32 cols,
                   const f64* __restrict__ Matrix){
    f64* __restrict__ tmp = aligned_alloc(64, sizeof(f64) * cols * rows);

    #pragma omp parallel for schedule(static)
    for(ui32 i = 0; i < rows; ++i){
        for(ui32 j = 0; j < cols; ++j)
            tmp[j * cols + i] = Matrix[i * cols + j];
    }

    return tmp;
}