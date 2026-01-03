#ifndef LLNN_NN_H
#define LLNN_NN_H
#include "matrix.h"
#include "activ.h"
#include "loss.h"
typedef struct neural_network {
	Matrix_t** weights;
	Matrix_t** biases;
	dfunc hidden_activ; // Input/hidden layer activation (f: double->double)
	mfunc output_activ; // Output layer activation (f: Matrix*->Matrix*)
	int n_layers;
} llnn_network_t;

llnn_network_t* ninit(int inputs, int hidden_layers, int hiddens, int outputs, dfunc hidden_activ, mfunc output_activ);
Matrix_t* npred(const llnn_network_t* nn, const Matrix_t* x, Matrix_t* out);
#endif