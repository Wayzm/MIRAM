#include "io.h"

double* read_matrix(const char* filename,
					const ui32 rows,
				    const ui32 cols){

	double* restrict matrix = aligned_alloc(64, sizeof(double) * rows * cols);
	FILE* matrix_file = fopen(filename, "r");

	if (!matrix_file){
        perror("Wrong filename or file not found. \n");
        exit(0);
    }
    int idx = 0;
    double number = 0;
    while (fscanf(matrix_file, "%lf", &number ) == 1 ){
    	matrix[idx] = number;
    	++idx;
    }

    return matrix;
}

void display_matrix(const double* matrix,
					const ui32 rows,
					const ui32 cols){

	for(ui32 i = 0; i < rows; ++i){
		for(ui32 j = 0; j < cols; ++j){
			printf("%lf ", matrix[i*cols + j]);
		}
		printf("\n");
	}
}
