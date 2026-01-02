#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

int mrealloc(Matrix_t *x, int rows, int cols){
	if(!x) return 1;
	if(x->rows != rows || x->cols != cols)
	{
		free(x->data);
		x->data = calloc(rows, cols * sizeof(double));
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
    // TODO: Better error handling, print when null is happening
	if(!a || !b) return NULL;
	if(a->cols != b->rows){
		return NULL;
	}

	// Matrix dimensions: (n x m) * (m x k) = (m x k)
	if(mrealloc(out, a->rows, b->cols) != 0) return NULL;


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

	if(mrealloc(out, a->rows, a->cols) != 0) return NULL;

	/* Set output matrix to the sum of the input matrices */
	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] + b->data[i];
	}
	return out;
}

Matrix_t* msub(const Matrix_t* a, const Matrix_t* b, Matrix_t* out){
	// Check conformability
	if(!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;

	if(mrealloc(out, a->rows, a->cols) != 0) return NULL;

	/* Set output matrix to the sum of the input matrices */
	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] - b->data[i];
	}
	return out;
}

// Frobenius norm
double mfrob(const Matrix_t* a){
	double sum = 0;

	if(!a)return 0.0;

	for(int i = 0; i < a->rows * a->cols; i++)
	{
		sum += a->data[i] * a->data[i];
	}

	return sqrt(sum);
}

Matrix_t* mrand(int rows, int cols, double min, double max, Matrix_t* out){
	// Allocate output matrix and check for null
	if(mrealloc(out, rows, cols) != 0) return NULL;
	if(!out)return NULL;

	/* Fill with random values */
	for(int i = 0; i < rows*cols; i++){
		double std_rand = (double)(rand() % 100000) / 100000.0;
		out->data[i] = (std_rand * (max - min)) + min;
	}

	return out;
}


Matrix_t* mscale(const Matrix_t* a, double b, Matrix_t* out){
	if(!a || !out) return NULL;
	if(mrealloc(out, a->rows, a->cols) != 0) return NULL;

	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] * b;
	}

	return out;
}


Matrix_t* mhad(const Matrix_t* a, const Matrix_t* b, Matrix_t* out){
	// Check conformability
	if(!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;

	if(mrealloc(out, a->rows, a->cols) != 0) return NULL;

	// Set output matrix to the Hadamard product of the input matrices
	for(int i = 0; i < a->rows*a->cols; i++){
		out->data[i] = a->data[i] * b->data[i];
	}
	return out;
}

Matrix_t* mtrns(const Matrix_t* a, Matrix_t* out){
	int row, col;

	if(!a)return NULL;
	if(mrealloc(out, a->cols, a->rows) != 0) return NULL;

	// Set output matrix to transposed input matrix
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[IDX_M(*out, col, row)] = a->data[IDX_M(*a, row, col)];
		}
	}

	return out;
}

Matrix_t* msel(int rows, int start, int stop){
	int out_dim;
	int cond;
	int row, col, row_idx;
	Matrix_t *sel_mat;
	// Sanity checks
	if (stop < start) return NULL;
	if (start > rows || stop > rows) return NULL;
	// The output dimension of the matrix is equal to the number of kept rows, the input is the number of original rows
	out_dim = stop - start;
	sel_mat = mnew(out_dim, rows);
	// Generate an identity matrix, except with the rows between start and stop kept only
	row_idx = 0;
	for(row = 0; row < rows; row++){
		// Only keep rows inside the slice list in normal mode, only keep rows outside in invert mode
		cond = row >= start && row < stop;
		// If this is a row to keep, create a proper identity row
		if(cond){
			for(col = 0; col < rows; col++){
				if(col == row) sel_mat->data[IDX_M(*sel_mat, row_idx, col)] = 1.0;
				else sel_mat->data[IDX_M(*sel_mat, row_idx, col)] = 0.0;
			}
			row_idx++;
		}
	}
	// Return the finished selection matrix
	return sel_mat;
}

Matrix_t* mslice(const Matrix_t* in, Matrix_t* out, int start, int stop, int t){
	Matrix_t *sel, *out_pre, *in_scaled;
	// Allocate
	out_pre = mnew(2, 2);
	in_scaled = mnew(1, 1);
	// Check for null
	if(in == NULL) return NULL;
	// Transpose the matrix if we are selecting rows and not columns
	if(t == 1) {
		mtrns(in, in_scaled);
	}
	else
	{
		mscale(in, 1.0, in_scaled);
	}
	// Sanity check on input
	if(stop > in_scaled->rows) stop = in_scaled->rows;
	// Generate the selection matrix*
	sel = msel(in_scaled->rows, start, stop);
	// Apply the selection matrix
	mmul(sel, in_scaled, out_pre);
	// Transpose back if necessary
	if(t == 1){
		mtrns(out_pre, out);
		mfree(out_pre);
		mfree(in_scaled);
	}
	else {
		// TODO: Implement mcopy function
		if(mrealloc(out, out_pre->rows, out_pre->cols) != 0) return NULL;
		for(int i = 0; i < out->rows*out->cols; i++){
			out->data[i] = out_pre->data[i];
		}
		out = out_pre;
	}
	// Free up variable
	mfree(sel);
	// Return the sliced matrix
	return out;
}
