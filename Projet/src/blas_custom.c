#include "blas_custom.h"

double GEVV_CLASSIC(const unsigned int size,
                      const double factor,
				      const double* vecteur_x,
				      const double* vecteur_y){

    double result = 0;

    #pragma omp parallel for schedule(dynamic, 1) reduction(+:result)
    for(size_t i = 0; i < size; ++i){
        result += vecteur_x[i] * vecteur_y[i];
    }
    return result;
}

double GEMV_CLASSIQUE(const unsigned int rows,
                      const unsigned int cols,
                      const double factor,
                      const double* __restrict__ matrix,
                      const double* __restrict__ vecteur);

double GEMV_MODIFIED(const unsigned int rows,
                     const unsigned int cols,
                     const double factor,
                     const double* __restrict__ matrix,
                     const double* __restrict__ vector);