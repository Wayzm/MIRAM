#include "blas_custom.h"

// vector_x * vector_y
double GEVV_CLASSIC(const unsigned int size,
                    const double factor,
				    const double* vector_x,
				    const double* vector_y){

    double result = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:result)
    for(size_t i = 0; i < size; ++i){
        result += factor * vector_x[i] * vector_y[i];
    }
    return result;
}

// factor * M * vector
double* GEMV_CLASSIQUE(const unsigned int rows,
                       const unsigned int cols,
                       const double factor,
                       const double* __restrict__ matrix,
                       const unsigned int size_vec,
                       const double* __restrict__ vector){

    double* __restrict__ result = aligned_alloc(64, sizeof(double) * size_vec);
    assert(cols == size_vec);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for(size_t i = 0; i < rows; ++i){
            for(size_t j = 0; j < size_vec; ++j)
                result[i] += factor * matrix[i * rows + j] * vector[j];
        }
    }
    
    return result;
}

double* GEMV_MODIFIED(const unsigned int rows,
                     const unsigned int cols,
                     const double factor,
                     const double* __restrict__ matrix,
                     const double* __restrict__ vector);