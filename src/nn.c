#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#include "nn.h"
#include "loss.h"
#include "activ.h"
#include "llnn.h"

llnn_network_t* ninit(int inputs, int hidden_layers, int hiddens, int outputs, dfunc hidden_activ, mfunc output_activ){
	llnn_network_t *nn;

	// Allocate all variables in the struct
	nn = malloc(sizeof(llnn_network_t));
	if(!nn){
        // TODO: Add error message
		return NULL;
	}
	nn->n_layers = hidden_layers + 2; // 1 Input layer + n hidden layers + 1 output layer
	// Allocate n_layers - 1 weights, as weights are applied between layers
	nn->weights = malloc((nn->n_layers - 1) * sizeof(Matrix_t*));
	nn->biases = malloc((nn->n_layers - 1) * sizeof(Matrix_t*));
	nn->hidden_activ = hidden_activ;
	nn->output_activ = output_activ;
	if(!nn->weights || !nn->biases){
		free(nn->weights);
		free(nn->biases);
		free(nn);
        // TODO: Freeing null?
        // TODO: add message for nulls
		return NULL;
	}

	// Allocate the individual weight and bias matrices
	// First weight maps R^inputs -> R^hiddens, so it should be hidden rows x input cols
    nn->weights[0] = mnew(1, 1);
	mrand(hiddens, inputs, -1.0, 1.0, nn->weights[0]);
	// First biases is added after transformation to R^hiddens, so it should be in R^hiddens
    nn->biases[0] = mnew(1, 1);
	mrand(hiddens, 1, -1.0, 1.0, nn->biases[0]);
	// Allocate the hidden layers and initialize them with ones. These map R^hiddens -> R^hiddens*/
	for(int i = 1; i < hidden_layers; i++){
        nn->weights[i] = mnew(1, 1);
        nn->biases[i] = mnew(1, 1);
		mrand(hiddens, hiddens, -1.0, 1.0, nn->weights[i]);
		mrand(hiddens, 1, -1.0, 1.0, nn->biases[i]);
	}
	// Allocate the output layer. This maps R^hiddens -> R^outputs, so it is outputs x hiddens
    nn->weights[hidden_layers] = mnew(1, 1);
	mrand(outputs, hiddens, -1.0, 1.0, nn->weights[hidden_layers]);
    DEBUG_PRINTF("Output weight layer (%d) dimensions: %d x %d, outputs: %d, hiddens: %d\n", hidden_layers, nn->weights[hidden_layers]->rows, nn->weights[hidden_layers]->cols, outputs, hiddens);
	// Last bias added to output, so it should be in R^outputs
    nn->biases[hidden_layers] = mnew(1, 1);
	mrand(outputs, 1, -1.0, 1.0, nn->biases[hidden_layers]);

	return nn;
}

Matrix_t* npred(const llnn_network_t* nn, const Matrix_t* x, Matrix_t* out){
	int layer;
	Matrix_t *current_vector, *product, *sum;
    sum = mnew(1, 1);
    product = mnew(1, 1);
    current_vector = mnew(1, 1);

	if(!nn || !x || !nn->weights || !nn->biases)return NULL;

	mscale(x, 1.0, current_vector);
	mprint(x);
	// There are 1 less weights than layers
	for(layer = 0; layer < nn->n_layers - 1; layer++){
		Matrix_t *res;
		// Apply the weights and biases
        DEBUG_PRINTF("---Size of weights on layer %d is %d x %d, weights are:\n", layer, nn->weights[layer]->rows, nn->weights[layer]->cols);
		DEBUG_MPRINT(nn->weights[layer]);
        // TODO: why is reallocation failing here
		res = mmul(nn->weights[layer], current_vector, product);
		DEBUG_PRINTF("Multiplication results:\n");
		DEBUG_MPRINT(res);
		DEBUG_PRINTF("Product dimensions: %d, %d, res: %p\n", product->rows, product->cols, (void *)res);
		madd(product, nn->biases[layer], sum);
        DEBUG_PRINTF("Sum dimensions: %d, %d\n", sum->rows, sum->cols);

		// Apply the activation function, if it exists, but not on the output layer
		if(nn->hidden_activ && layer < nn->n_layers-1){
			DEBUG_PRINTF("Mscale NOT applying...\n");
			mapply(sum, nn->hidden_activ, current_vector);
		}
		else{
			DEBUG_PRINTF("Mscale applying..\n");
			// TODO: Why is mscale() switching dimensions?
			mscale(sum, 1.0, current_vector);
		}
		DEBUG_PRINTF("Final current_vector dimensions: %d x %d\n", current_vector->rows, current_vector->cols);
		DEBUG_MPRINT(current_vector);
	}

	// Apply output activation, if applicable
	if(nn->output_activ){
		sum = nn->output_activ(current_vector);
		mfree(current_vector);
		current_vector = sum;
	}
    DEBUG_PRINTF("current_vector dimensions: %d, %d\n", current_vector->rows, current_vector->cols);

    // TODO: Free current_vector better
    mfree(sum);
    mfree(product);

	// Return the final predicted column vector
    if(mrealloc(out, current_vector->rows, current_vector->cols) != 0) return NULL;
    for(int i = 0; i < current_vector->rows * current_vector->cols; i++)
    {
        out->data[i] = current_vector->data[i];
    }
	return current_vector;
}


Matrix_t* ndiff(const Matrix_t* x, const dfunc activ_func, Matrix_t *d_activ){
	double h = 0.000001;
	Matrix_t *activ_x, *activ_xh, *xh, *h_mat;

	if(!x || !activ_func) return NULL;

	h_mat = mnew(x->rows, x->cols);
	xh = mnew(x->rows, x->cols);
	activ_x = mnew(x->rows, x->cols);
	activ_xh = mnew(x->rows, x->cols);
    if(mrealloc(d_activ, x->rows, x->cols) != 0) return NULL;
	// TODO: check nulls

	// Vector of x + h
	for(int i = 0 ; i < x->rows * x->cols; i++)
	{
		h_mat->data[i] = h;
	}
	madd(x, h_mat, xh);

	/* Numerically calculate derivative of activ_func wrt x.
	d_activ = (f(x+h)-f(x))/h */
	// Numerator (f(x+h)-f(x))
	mapply(x, activ_func, activ_x);
	mapply(xh, activ_func, activ_xh);
	msub(activ_xh, activ_x, d_activ);
	// Denominator (divide by h)
	mscale(d_activ, 1.0/h, d_activ);

	mfree(xh);
	mfree(activ_xh);
	mfree(activ_x);
	mfree(h_mat);
	return d_activ;
}