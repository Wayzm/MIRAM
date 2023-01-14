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
f64* GEMV_CLASSIQUE(const ui32 rows,
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
                      const f64* __restrict__ matrix,
                      const ui32 size_vec,
                      const f64* __restrict__ vector){
    assert(rows == size_vec);
    f64 result = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:result)
    for(ui32 i = 0; i < size_vec; ++i)
        for(ui32 j = 0; j < rows; ++j)
            result += factor * matrix[j * cols + i] * vector[i];

    return result;
}