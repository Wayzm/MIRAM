#include "blas_custom.h"

// vector_x * vector_y
f64 GEVV_CLASSIC(const ui32 size,
                 const f64 factor,
				 const f64* vector_x,
				 const f64* vector_y){

    f64 result = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:result)
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
    f64* __restrict__ result = aligned_alloc(64, sizeof(f64) * size_vec);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(ui32 i = 0; i < rows; ++i){
            for(ui32 j = 0; j < size_vec; ++j)
                result[i] += factor * matrix[i * rows + j] * vector[j];
        }
    }
    
    return result;
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
        result += factor * matrix[increment * cols + i] * vector[i];

    return result;
}

// factor * A[i: 0...N] * B[j:0...N]
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
		for(ui32 j = 0U; j < cols_B; ++j){
			result[i] += matrix_A[i * cols_A + increment_A] * matrix_B[i * cols_B + j];
		}
	}
    return result;
}