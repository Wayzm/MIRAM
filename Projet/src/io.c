#include "io.h"

double* read_matrix(const char* filename, 
					const unsigned int rows,
				    const unsigned int cols){

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
					const unsigned int rows,
					const unsigned int cols){

	for(unsigned int i = 0U; i < rows; ++i){
		for(unsigned int j = 0U; j < cols; ++j){
			printf("%lf ", matrix[i*cols +j]);
		}
		printf("\n");
	}
}
