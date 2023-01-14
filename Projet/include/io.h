#ifndef IO_H
#define IO_H

#include "lib.h"
/***
 * read_matrix will create a matrix from an input file
 * display_matrix will display on the pty the input matrix
 ***/
f64* read_matrix(const char* filename,
					   const ui32 rows,
					   const ui32 cols);

void display_matrix(const f64* matrix,
					const ui32 rows,
					const ui32 cols);

#endif