#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"



Matrix_t* mnew(int rows, int cols){
	Matrix_t *new_matrix;
	int row;

	new_matrix = malloc(sizeof(Matrix_t));
	if(!new_matrix)return NULL;
	new_matrix->rows = rows;
	new_matrix->cols = cols;

	new_matrix->data = malloc(rows * cols * sizeof(double));
	if(!new_matrix->data){
		free(new_matrix->data);
		free(new_matrix);
		return NULL;
	}
	return new_matrix;
}

void mfree(Matrix_t* x){
	int row;
	if(!x) return;
	free(x->data);
	free(x);
}

Matrix_t* mmul(const Matrix_t* a, const Matrix_t* b, Matrix_t *out){
	int row, col, index;

	// Check for nulls and conformability
    // TODO: Better error handling
	if(!a || !b) return NULL;
	if(a->cols != b->rows){
		return NULL;
	}

	// Matrix dimesions: (n x m) * (m x k) = (m x k)
    if(out->rows != a->rows || out->cols != b->cols)return NULL;
	if(!out)return NULL;

	//For each row in matrix a
	for(row = 0; row < a->rows; row++){
		// For each column matrix b
		for(col = 0; col < b->cols; col++){
			/* Set the output cell to the sum of the products of the entries in the row of a
			and the column of b. */
            // TODO: Add indexer macro
			out->data[IDX_M(*out, row, col)] = 0;
			for(index = 0; index < a->cols; index++){
                out->data[IDX_M(*out, row, col)] += a->data[IDX_M(*a, row, index)] * b->data[IDX_M(*out, index, col)];
			}
		}
	}

	return out;
}