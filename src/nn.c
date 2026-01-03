#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#include "nn.h"
#include "loss.h"
#include "activ.h"

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
    printf("Output weight layer (%d) dimensions: %d x %d, outputs: %d, hiddens: %d\n", hidden_layers, nn->weights[hidden_layers]->rows, nn->weights[hidden_layers]->cols, outputs, hiddens);
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
	// There are 1 less weights than layers
	for(layer = 0; layer < nn->n_layers - 1; layer++){
		Matrix_t *res;
		mfree(product);
		product = mnew(1,1);
		// Apply the weights and biases
        printf("Size of weights on layer %d is %d x %d\n", layer, nn->weights[layer]->rows, nn->weights[layer]->cols);
        // TODO: why is reallocation failing here
		res = mmul(nn->weights[layer], current_vector, product);
		printf("Product dimensions: %d, %d, res: %p\n", product->rows, product->cols, (void *)res);
		madd(product, nn->biases[layer], sum);
        printf("Sum dimensions: %d, %d\n", sum->rows, sum->cols);

		// Apply the activation function, if it exists, but not on the output layer
		if(nn->hidden_activ && layer < nn->n_layers-1){
			mapply(sum, nn->hidden_activ, current_vector);
		}
		else{
			mscale(sum, 1.0, current_vector);
		}
	}

	// Apply output activation, if applicable
	if(nn->output_activ){
		sum = nn->output_activ(current_vector);
		mfree(current_vector);
		current_vector = sum;
	}
    printf("current_vector dimensions: %d, %d\n", current_vector->rows, current_vector->cols);

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