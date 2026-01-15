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


Matrix_t*** nbprop(const llnn_network_t* nn, const Matrix_t* X_train, const Matrix_t* y_train, const lfunc loss_func,
				 const lfuncd dloss_func){
	/* http://neuralnetworksanddeeplearning.com/chap2.html#the_code_for_backpropagation */
	/* Nabla_b and nabla_w are gradients of the biases and weights respectively. They are lists of Matrices
	just as the weights and biases are in the neural network structure */
	Matrix_t **nabla_b, **nabla_w;
	Matrix_t*** nablas; /* Holds both the weight and bias gradients */
	Matrix_t **Zs; /* A list of Z vectors (unactivated outputs) for each layer */
	Matrix_t **activations = NULL; /* A list of activations for each layer */
	Matrix_t *z = NULL; /* Current z (unactivated layer output) vector */
	Matrix_t *activation = NULL; /* Current activation */
	Matrix_t *activationp; /* Activation prime */
	Matrix_t *last_activation; /* Activation of last layer */
	Matrix_t *err = mnew(1,1); /* Error (output of loss function) */
	Matrix_t *delta; /* Delta for current layer */
	Matrix_t *tmp = NULL, *tmp2 = NULL; /* Temporary variables for calculations */
	int layer, layer_fwd;
	size_t list_size;

	/* Check for nulls */
	if(!nn || !X_train || !y_train || !loss_func) return NULL;
	/* Make sure we only have 1 row of data */
	if(X_train->cols != 1 || y_train->cols != 1){
		DEBUG_PRINTF("nbprop(): X_train or Y_train do not 1 col");
		return NULL;
	}
	/* Make sure training data is right size */
	/* For matrix multiplication, rows of the first matrix must be equal to cols of the second. Weight*Activation */
	if(X_train->rows != nn->weights[0]->cols){
		DEBUG_PRINTF("nbprop(): X_train rows does not equal weights[0] cols");
		mprint(X_train);
		mprint(nn->weights[0]);
		return NULL;
	}

	/* Allocate variables */
	list_size = nn->n_layers * sizeof(Matrix_t*);
	/* nabla_b = [np.zeros(b.shape) for b in self.biases] */
	nabla_b = malloc(list_size);
	/* nabla_w = [np.zeros(w.shape) for w in self.weights] */
	nabla_w = malloc(list_size);
	/* zs = [] */
	Zs = calloc(list_size + 1, sizeof(Matrix_t*));
	/* activations = [x] */
	activations = calloc(list_size + 1, sizeof(Matrix_t*));

	/* Get data from designated row of X_train and y_train for stochastic gradient descent */
	/*
	MDUP(&X_train->data[index], cur_X_train, 1, X_train->cols);
	MDUP(&y_train->data[index], cur_y_train, 1, y_train->cols);
	*/

	/* Set initial activation to the row of training data (cur_X_train we are training on */
	/* activation = x */
	activation = mnew(1, 1);
	mscale(X_train, 1.0, activation); // Was NULL
	activations[0] = activation;

	/* Run the forward propagation (prediction) pass. There are n_layers - 1 weights/biases in the network*/
	/* for b, w in zip(self.biases, self.weights): */
	
	for(layer = 0; layer < nn->n_layers - 1; layer++){
		Matrix_t *weight, *bias;
		bias = nn->biases[layer];
		weight = nn->weights[layer];
		
		
		
		
		
		
		
		

		/* Calculate Z (unactivated layer output) */
		/* z = np.dot(w, activation)+b */
		tmp = mnew(1, 1);
		z = mnew(1, 1);
		mmul(weight, activation, z); // Was NULL
		madd(tmp, bias, z); // Addition is not in place
		
		
		/* zs.append(z)  */
		Zs[layer] = z;

		/* Calculate the activation by applying it to Z (the output of the layer before activtion) */
		/* activation = sigmoid(z) */
		activation = mnew(1, 1);
		mapply(z, nn->hidden_activ, activation);
		/* activations.append(activation) */
		activations[layer + 1] = activation;
		
		mfree(tmp);
	}

	/* Calculate output delta*/
	/* delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) */
	delta = dloss_func(activation, y_train); /* Derviative of loss function wrt output activations */
	
	
	
	
	
	
	
	
	
	ndiff(Zs[nn->n_layers - 2], nn->hidden_activ, err); /* Derivative of activation function wrt output Z vector */
	/*err = mapply(Zs[nn->n_layers - 2], &dsigm, NULL);*/
	
	
	delta = mhad(delta, err, delta); /* Delta is the Hadamard product of these 2, equation BP1 */
	
	

	/* Calculate output weight and bias derivatives */
	/* Definition of dot product: x.y=x^T*y */
	last_activation = mnew(1, 1);
	mtrns(activations[nn->n_layers - 2], last_activation); // Was null
	/* nabla_b[-1] = delta */
	/* In a 4 layer network, this would be nabla_b[3] */
	/* Last element of nabla_b is nn->n_layers - 1 and not nn->n_layers */
	nabla_b[nn->n_layers - 2] = mnew(1, 1);
	mscale(delta, 1, nabla_b[nn->n_layers - 2]); /* Equation BP3 */ // Was null
	/* nabla_w[-1] = np.dot(delta, activations[-2].transpose()) */
	nabla_w[nn->n_layers - 2] = mnew(1, 1);
	mmul(delta, last_activation, nabla_w[nn->n_layers - 2]); /* Equation BP4 */ // Was null

	mfree(err);
	mfree(tmp);
	mfree(last_activation);

	/*  for l in xrange(2, self.num_layers): */
	/* In 4 layer network:
	weights[0]
	weights[1]
	weights[2]
	weights[3]
	nn->n_layers = 4
	weights[-1] should be weights[3]
	weights[-2] should be weights[2]
	So, the layer_fwd should go 2, 3 which corresponds to layers 2, 1
	*/
	
	for(layer_fwd = 2; layer_fwd < nn->n_layers; layer_fwd++){
		tmp = mnew(1, 1);
		int layer = nn->n_layers - layer_fwd; /* Account for only n_layers - 1 weights, but also add 1 */
		Matrix_t *transposed_weights;
		
		
		
		
		
		
		/*  z = zs[-l] */
		z = Zs[layer - 1]; /* Z vector for current layer (unactivated layer output) */
		/* sp = sigmoid_prime(z) */
		activationp = mnew(1, 1);
		ndiff(z, nn->hidden_activ, activationp); /* Derivative of activation function for current layer */
		/*activationp = mapply(z, drelu, NULL);*/
		/* last_activation = activations[-l-1].transpose() */
		last_activation = mnew(1, 1);
		activations[layer - 1] = mnew(1, 1); // ?
		mtrns(activations[layer - 1], last_activation); /* Transpose of activation of layer n-1 */ // Was null
		
		
		
		
		
		
		
		

		/* Calculate delta */
		/* delta = np.dot(self.weights[-l+1].transpose(), delta) * sp */
		/* tmp = np.dot(self.weights[-l+1].transpose(), delta) */
		transposed_weights = mnew(1,1);
		mtrns(nn->weights[layer], transposed_weights); /* Transposed weights of next layer */ // Was null
		mmul(transposed_weights, delta, tmp);
		/* delta = tmp * sp */
		
		
		
		
		
		
		mfree(delta);
		delta = mnew(1, 1);
		mhad(tmp, activationp, delta); /* Equation BP2 */ // Was null
		
		

		/* Calculate gradients */
		/* nabla_b[-l] = delta */
		nabla_b[layer - 1] = mnew(1, 1);
		mscale(delta, 1.0, nabla_b[layer - 1]); /* Equation BP3 */ // Was null
		
		
		/* nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) */
		nabla_w[layer - 1] = mnew(1, 1);
		mmul(delta, last_activation, nabla_w[layer - 1]); /* Equation BP4 */
		
		
		
		
		
		
		

		/* Free variables */
		mfree(transposed_weights);
		mfree(last_activation);
		mfree(activationp);
		mfree(tmp);
	}


	/* Free unneeded variables */
	for(layer = 0; layer < nn->n_layers; layer++){
		mfree(Zs[layer]);
		
		
		
		mfree(activations[layer]);
	}
	mfree(tmp2);
	mfree(delta);
	free(activations);
	free(Zs);

	/* Package up and return pointer to gradients */
	nablas = malloc(2 * sizeof(Matrix_t**));
	nablas[0] = nabla_w;
	nablas[1] = nabla_b;
	return nablas;
}