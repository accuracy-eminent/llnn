#include <math.h>
#include "matrix.h"
#include "activ.h"

/* ReLU (rectified linear unit) */
double arelu(double x){
	return (x >= 0) ? x : 0;
}

// Derivative
double drelu(double output){
	return output > 0;
}

// Leaky ReLU
double alrelu(double x){
	return (x >= 0) ? x : 0.01*x;
}

// Linear function is simply the identity function
double alin(double x){
	return x;
}

// Sigmoid activation function
double asigm(double x){
	return 1/(1 + exp(-1*x));
}

// Derivative of sigmoid
double dsigm(double output){
	return asigm(output) * (1.0 - asigm(output));
}

// Tanh activation function
double atanh(double x){
	return (exp(x) - exp(-1*x)) / (exp(x) + exp(-1*x));
}

// Softmax function, used for estimating probabilities from raw outputs
Matrix_t* asmax(const Matrix_t* a){
	Matrix_t* out;
	int row, col;
	double sum = 0.0;

	out = mnew(a->rows, a->cols);

	/* Calculate exp(x) for each x in the matrix a, and update the sum */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[IDX_M(*out, row, col)] = exp(a->data[IDX_M(*a, row, col)]);
			sum += out->data[IDX_M(*out, row, col)];
		}
	}

	/* Scale each entry by the sum */
	for(row = 0; row < a->rows; row++){
		for(col = 0; col < a->cols; col++){
			out->data[IDX_M(*out, row, col)] /= sum;
		}
	}

	return out;
}
