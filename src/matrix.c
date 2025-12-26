#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

int mrealloc(Matrix_t *x, int rows, int cols){
	if(x->rows != rows || x->cols != cols)
	{
		free(x->data);
		x->data = malloc(rows * cols * sizeof(double));
		if(!x->data)
		{
			return 1;
		}
		x->rows = rows;
		x->cols = cols;
	}
	return 0;
}

int mcmp(const Matrix_t* a, const Matrix_t* b){
	// If matrices aren't the same dimensions they can't be equal
	if(!a || !b || a->rows != b->rows || a->cols != b->cols) return 0;

	// If any cell is not equal, then matrices are not equal
	for(int i = 0; i < a->rows * a->cols; i++)
	{
		if(a->data[i] != b->data[i]) return 0;
	}

	// Otherwise, if all cells are equal, the matrices are equal
	return 1;
}

Matrix_t* mnew(int rows, int cols){
	Matrix_t *new_matrix;

	new_matrix = malloc(sizeof(Matrix_t));
	if(!new_matrix)return NULL;
	new_matrix->rows = rows;
	new_matrix->cols = cols;

	new_matrix->data = calloc(rows, cols * sizeof(double));
	if(!new_matrix->data){
		free(new_matrix->data);
		free(new_matrix);
		return NULL;
	}
	return new_matrix;
}

void mfree(Matrix_t* x){
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
	// TODO: Auto-reallocate the matrix
	if(!out)return NULL;
    if(out->rows != a->rows || out->cols != b->cols)return NULL;

	//For each row in matrix a
	for(row = 0; row < a->rows; row++){
		// For each column matrix b
		for(col = 0; col < b->cols; col++){
			/* Set the output cell to the sum of the products of the entries in the row of a
			and the column of b. */
			out->data[IDX_M(*out, row, col)] = 0;
			for(index = 0; index < a->cols; index++){
                out->data[IDX_M(*out, row, col)] += a->data[IDX_M(*a, row, index)] * b->data[IDX_M(*b, index, col)];
			}
		}
	}

	return out;
}

Matrix_t* madd(const Matrix_t* a, const Matrix_t* b, Matrix_t* out){
	// Check conformability
	if(!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;

	// TODO: Auto-adjust matrix dimensions
	if(!out) return NULL;
	if(out->rows != a->rows || out->cols != a->cols)return NULL;

	/* Set output matrix to the sum of the input matrices */
	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] + b->data[i];
	}
	return out;
}

Matrix_t* mscale(const Matrix_t* a, double b, Matrix_t* out){
	if(!a || !out) return NULL;
	if(out->rows != a->rows || out->cols != a->cols) return NULL;

	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] * b;
	}

	return out;
}