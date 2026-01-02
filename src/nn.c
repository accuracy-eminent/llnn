#include <stdlib.h>
#include <math.h>
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
	// Last bias added to output, so it should be in R^outputs
    nn->biases[hidden_layers] = mnew(1, 1);
	mrand(outputs, 1, -1.0, 1.0, nn->biases[hidden_layers]);

	return nn;
}