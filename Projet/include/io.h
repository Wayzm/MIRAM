#ifndef IO_H
#define IO_H

#include "lib.h"
/***
 * read_matrix will create a matrix from an input file
 * display_matrix will display on the pty the input matrix
 * 
 * 
 ***/
double* read_matrix(const char* filename,
					   const unsigned int rows,
					   const unsigned int cols);

void display_matrix(const double* matrix,
					const unsigned int rows,
					const unsigned int cols);

#endif