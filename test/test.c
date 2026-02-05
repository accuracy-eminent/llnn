#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "minunit.h"
#include "../src/matrix.h"
#include "../src/activ.h"
#include "../src/loss.h"
#include "../src/nn.h"
#include "../src/io.h"

int tests_run = 0;

static char* test_mnew(){
	Matrix_t* mat;
	mat = mnew(5,3);
	mu_assert("Error, rows != 5", mat->rows == 5);
	mu_assert("Error, cols != 3", mat->cols == 3);
	mfree(mat);
	return NULL;
}

static char* test_mcmp(){
	Matrix_t *a, *b;
	double a_data[4] = {
		1.0, 0.0,
		0.0, 1.0
	};
	double b_data[6] = {
		4.0, 4.0, 3.0,
		1.0, 1.0, 2.0
	};
	a = mnew(2, 2);
	b = mnew(2, 3);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));

	mu_assert("Error, a != a", mcmp(a, a));
	mu_assert("Error, a == b", !mcmp(a, b));
	mu_assert("Error, b != b", mcmp(b, b));

	mfree(a);
	mfree(b);

	return NULL;
}

static char* test_mmul(){
	Matrix_t *a, *b, *c, *prod;
	double a_data[4] = {
		1., 2.,
		2., 1.
	};
	double b_data[6] = {
		2., 3., 4.,
		5., 6., 7.
	};
	double c_data[6] = {
		12., 15., 18.,
		9., 12., 15.
	};

	// Convert arrays into matrices
	a = mnew(2,2);
	b = mnew(2,3);
	c = mnew(2,3);
	prod = mnew(2, 3);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));
	memcpy(c->data, c_data, sizeof(c_data));


	// Run tests
	mmul(a, b, prod);
	mu_assert("Error, a * b != c", mcmp(prod, c));
	mfree(prod);
	prod = mmul(b, a, prod);
	mu_assert("Error, b * a != NULL", !prod);
	mfree(prod);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);

	return NULL;
}

static char* test_madd(){
	Matrix_t *a, *b, *c, *sum;
	double a_data[4] = {
		1., 2.,
		2., 1.
	};
	double b_data[4] = {
		3., 4.,
		5., 6.
	};
	double c_data[4] = {
		4., 6.,
		7., 7.
	};

	// Convert arrays into matrices
	a = mnew(2,2);
	b = mnew(2,2);
	c = mnew(2,2);
	sum = mnew(2, 2);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));
	memcpy(c->data, c_data, sizeof(c_data));

	// Run tests
	madd(a, b, sum);
	mu_assert("Error, a + b != c", mcmp(sum, c));
	mfree(sum);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);

	return NULL;
}

static char* test_mscale(){
	Matrix_t *a, *as, *b, *c, *cs, *d;
	double a_data[4] = {
		1.0, 2.0,
		3.0, 4.0
	};
	double b_data[4] = {
		2.0, 4.0,
		6.0, 8.0
	};

	as = mnew(2, 2);
	cs = mnew(1, 1);

	a = mnew(2, 2);
	b = mnew(2, 2);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));

	c = mnew(1, 1);
	d = mnew(1, 1);
	c->data[0] = -8.0;
	d->data[0] = -8.0;

	mscale(a, 2.0, as);
	mu_assert("Error, b != mscale(a)", mcmp(b, as));
	mscale(c, 1.0, cs);
	mu_assert("Error, d != mscale(c)", mcmp(d, cs));

	mfree(as);
	mfree(cs);
	mfree(a);
	mfree(b);
	mfree(c);
	mfree(d);

	return NULL;
}

static char* test_mhad(){
	Matrix_t *a, *b, *c, *prod;
	double a_data[4] = {
		1., 2.,
		2., 1.
	};
	double b_data[4] = {
		3., 4.,
		5., 6.
	};
	double c_data[4] = {
		3., 8.,
		10., 6.
	};

	// Convert arrays into matrices
	a = mnew(2,2);
	b = mnew(2,2);
	c = mnew(2,2);
	prod = mnew(2, 2);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));
	memcpy(c->data, c_data, sizeof(c_data));

	// Run tests
	mhad(a, b, prod);
	mu_assert("Error, a x b != c", mcmp(prod, c));
	mfree(prod);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(c);

	return NULL;
}

static char* test_mtrns(){
	Matrix_t *a, *b, *at;
	double a_data[6] = {
		1., 2., 3.,
		4., 5., 6.
	};
	double b_data[6] = {
		1., 4.,
		2., 5.,
		3., 6.
	};

	// Convert arrays into matrices
	a = mnew(2,3);
	b = mnew(3,2);
	at = mnew(3,2);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));

	// Run tests
	mtrns(a, at);
	mu_assert("Error, transposed a != b", mcmp(at, b));

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(at);

	return NULL;
}


static char* test_mslice(){
	Matrix_t *a, *b, *out;
	double a_data[6] = {
		1., 2., 3.,
		4., 5., 6.
	};
	double b_data[6] = {
		1., 4.,
		2., 5.,
		3., 6.
	};

	// Convert arrays into matrices
	a = mnew(2,3);
	b = mnew(3,2);
	out = mnew(1,1);
	memcpy(a->data, a_data, sizeof(a_data));
	memcpy(b->data, b_data, sizeof(b_data));

	// Run tests
	mslice(b, out, 0, 2, 0);
	mu_assert("Sliced size is wrong!", out->rows == 2 && out->cols == 2);
	mu_assert("Values are wrong!", (fabs(out->data[3] - 5.0f) < 0.1) && fabs(out->data[0] - 1.0f) < 0.1);
	for(int i = 0; i < 4; i++)
	{
		out->data[i] = 0.0;
	}
	mslice(a, out, 0, 2, 1);
	mu_assert("Sliced size is wrong!", out->rows == 2 && out->cols == 2);
	mu_assert("Values are wrong!", (fabs(out->data[3] - 5.0f) < 0.1) && fabs(out->data[0] - 1.0f) < 0.1);

	// Free variables
	mfree(a);
	mfree(b);
	mfree(out);

	return NULL;
}


static char* test_mfrob(){
	Matrix_t *a;
	a = mnew(3, 1);
	a->data[0] = 3.0;
	a->data[1] = 4.0;
	double norm = mfrob(a);
	mu_assert("Norm != 5", fabs(norm - 5.0) < 0.1);
	mfree(a);
	return NULL;
}

static char* test_mrand()
{
	Matrix_t *a;
	int plus_count = 0;
	int minus_count = 0;
	a = mnew(3, 3);
	mrand(3, 3, 1.0, 2.0, a);
	mu_assert("Values are not in the correct range!", a->data[0] >= 1.0 && a->data[0] <= 2.0 && a->data[1] >= 1.0 && a->data[1] <= 2.0 && a->data[0] != a->data[1]);
	mfree(a);
	// Test positive and negative values
	a = mnew(2, 10);
	mrand(2, 10, -1.0, 1.0, a);
	for(int i = 0; i < a->rows*a->cols; i++){
		if(a->data[i] < 0)minus_count++;
		else if(a->data[i] > 0)plus_count++;
	}
	mu_assert("Mrand() not producing both positive and negative numbers!\n", plus_count > 0 && minus_count > 0);
	return NULL;
}

static char* test_asmax(){
	Matrix_t *a, *smax;
	a = mnew(3, 1);
	a->data[0] = 1.0;
	a->data[1] = 2.0;
	a->data[2] = 3.0;
	smax = asmax(a);
	mu_assert("Output size is wrong!", smax->cols == a->cols && smax->rows == a->rows);
	mu_assert("Softmax values do not sum to 1!\n", fabs(smax->data[0] + smax->data[1] + smax->data[2] - 1.0) < 0.01);
	mfree(a);
	mfree(smax);
	return NULL;
}

static char* test_lmse()
{
	Matrix_t *a, *b;
	a = mnew(1, 3);
	b = mnew(1, 3);
	a->data[0] = 1.0;
	a->data[1] = 1.0;
	a->data[2] = 1.0;
	b->data[0] = 0.0;
	b->data[1] = 0.0;
	b->data[2] = 0.0;
	// double lmse(const Matrix_t* actual, const Matrix_t* pred);
	double res = lmse(a, b);
	mu_assert("MSE != 1.0", res == 1.0);
	return NULL;
}

static char* test_ninit()
{
	llnn_network_t *nn = ninit(2, 2, 4, 2, &arelu, NULL);
	mu_assert("Weights are NULL", nn->weights != NULL);
	mu_assert("Biases are NULL", nn->biases != NULL);
	mu_assert("First layer should be hiddens x inputs", nn->weights[0]->rows == 4 && nn->weights[0]->cols == 2);
	return NULL;
}

static char* test_npred()
{
	Matrix_t *in, *out;
	in = mnew(2, 1);
	in->data[0] = 6;
	in->data[0] = 5;
	out = mnew(1, 1);
	llnn_network_t *nn = ninit(2, 2, 4, 2, &asigm, NULL);
	npred(nn, in, out);
	printf("Out dimensions: %d, %d\n", out->cols, out->rows);
	mu_assert("Output size is wrong", out->cols == 1 && out->rows == 2);
	printf("Out values: %f, %f\n", out->data[0], out->data[1]);
	// TODO: More extensive accurate results in testing, configure manually with identity matrix, etc
	return NULL;
}

static char* test_ndiff()
{
	// For reLU, f'(x) will be 1 if x>0, 0 otherwise 
	Matrix_t *x, *xd;
	x = mnew(2, 2);
	xd = mnew(2, 2);
	x->data[0] = 5;
	x->data[1] = -20;
	x->data[2] = 30;
	x->data[3] = 4;

	ndiff(x, &arelu, xd);

	mu_assert("xd->0 != 1", fabs(xd->data[0] - 1.0) < 0.1);
	mu_assert("xd->1 != 0", fabs(xd->data[1] - 0.0) < 0.1 );
	mu_assert("xd->2 != 1", fabs(xd->data[0] - 1.0) < 0.1);
	mu_assert("xd->3 != 1", fabs(xd->data[0] - 1.0) < 0.1);
	
	mfree(x);
	mfree(xd);
	return NULL;
}

static char* test_nbprop()
{
	// Matrix_t*** nbprop(const llnn_network_t* nn, const Matrix_t* X_train, const Matrix_t* y_train, const lfunc loss_func, const lfuncd dloss_func){
	Matrix_t *xt, *yt;
	Matrix_t ***nablas;

	// TODO: Get dimensions right
	xt = mnew(2, 1);
	xt->data[0] = 1;
	xt->data[1] = 2;
	yt = mnew(2, 1);
	yt->data[0] = 3;
	yt->data[1] = 5;
	// input is 4 wide, 2 hidden layers, hidden layers are 4 wide, output is 2 wide 
	llnn_network_t *nn = ninit(2, 2, 4, 2, &asigm, NULL);
	nablas = nbprop(nn, xt, yt, lmse, dmse);
	// For now, check that we don't have all zeroes
	mu_assert("All zeroes in weights in nbprop()!", nablas[0][0]->data[0] != 0 || nablas[0][0]->data[1] != 0);
	mu_assert("All zeroes in biases in nbprop()!", nablas[1][0]->data[0] != 0 || nablas[1][0]->data[1] != 0);
	// Check dimensions
	mu_assert("1st nabla should be 2 input layer -> 4 hidden layer 1", nablas[0][0]->cols == 2 && nablas[0][0]->rows == 4);
	mu_assert("2nd nabla should be 4 hidden layer 1 -> 4 hidden layer 2", nablas[0][1]->cols == 4 && nablas[0][1]->rows == 4);
	mu_assert("2nd nabla should be 4 hidden layer 2 -> 2 output layer", nablas[0][2]->cols == 4 && nablas[0][2]->rows == 2);
	mu_assert("1st nabla bias should be 4 hidden layer 1", nablas[1][0]->cols == 1 && nablas[1][0]->rows == 4);
	mu_assert("2nd nabla bias should be 4 hidden layer 2", nablas[1][1]->cols == 1 && nablas[1][1]->rows == 4);
	mu_assert("2nd nabla bias should be 2 output layer", nablas[1][2]->cols == 1 && nablas[1][2]->rows == 2);
	return NULL;
}

static char* test_ntrain()
{
	srand(42);
	// Set up training data, y=2x+1 with error of +/- 0.2
	Matrix_t *x = mnew(8, 1);
	//double x_data[8] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
	double x_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	memcpy(x->data, x_data, sizeof(x_data));
	Matrix_t *y = mnew(8, 1);
	for(int i = 0; i < 8; i++)
	{
		y->data[i] = (x->data[i] * 2.0) + 1.0 + ((i % 2)? 0.2: -0.2);
	}
	
	// Train the network
	llnn_network_t* nn = ninit(1, 2, 8, 1, &alrelu, NULL);
	ntrain(nn, x, y, lmse, dmse, 1900, 0.00001);

	// Predict and get the mean squared error
	Matrix_t *preds = mnew(x->rows, x->cols);
	for(int i = 0; i < x->rows; i++)
	{
		Matrix_t *pred = mnew(1, 1);
		Matrix_t *x_in = mnew(1, 1);
		x_in->data[0] = x->data[i];
		npred(nn, x_in, pred);
		preds->data[i] = pred->data[0];
		mfree(pred);
		mfree(x_in);
	}
	float mse = lmse(y, preds);
	printf("PREDS:\n");
	mprint(preds);
	printf("Y:\n");
	mprint(y);

	printf("MSE: %f\n", mse);

	return NULL;
}

static char* test_npredm()
{
	Matrix_t *in, *out;
	in = mnew(2, 2);
	in->data[0] = 6;
	in->data[1] = 5;
	in->data[2] = 1;
	in->data[3] = 2;
	out = mnew(1, 1);
	llnn_network_t *nn = ninit(2, 2, 4, 2, &asigm, NULL);
	npredm(nn, in, out);
	// TODO: Add more input dimensions
	mu_assert("Output size is wrong", out->cols == 2 && out->rows == 2);
	return NULL;
}

static char* test_io(){
	Matrix_t *iris;
	iris = mnew(1, 1);
	ireadcsv("datasets/iris2.csv", iris);
	mu_assert("Error, could not load in CSV data!", iris != NULL);
	mu_assert("Number of columns is wrong", iris->cols == 7);
	mfree(iris);
	return NULL;
}

// Test categorical classification of iris dataset
static char* test_categorical(){
	Matrix_t *iris, *iris_train, *iris_test, *X_train, *y_train, *X_test, *y_test;
	Matrix_t *preds, *preds2, *preds_vec, *y_vec, *conv_vec;
	llnn_network_t *nn;
	double acc;

	/* Load in data, split into train and test */
	iris = mnew(1, 1);
	ireadcsv("datasets/iris2.csv", iris);
	mu_assert("Error, could not load in CSV data!", iris != NULL);
	iris_train = mnew(1, 1);
	mslice(iris, iris_train, 0, 50, 0);
	iris_test = mnew(1, 1);
	mslice(iris, iris_test, 50, 60, 0);
	X_train = mnew(1, 1);
	mslice(iris_train, X_train, 0, 4, 1);
	y_train = mnew(1, 1);
	mslice(iris_train, y_train, 4, 7, 1);
	X_test = mnew(1, 1);
	mslice(iris_test, X_test, 0, 4, 1);
	y_test = mnew(1, 1);
	mslice(iris_test, y_test, 4, 7, 1);


	/* Initialize and train the network on the training set */
    nn = ninit(4, 2, 16, 3, &alrelu, &asmax);
    ntrain(nn, X_train, y_train, lmse, dmse, 5000, 0.001);


    /* Make and test predictions */
	preds = mnew(1, 1);
    npredm(nn, X_test, preds);
	preds2 = ahmax(preds);

	/* Convert from one-hot to categorical (1,2,3) */
	conv_vec = mnew(3, 1);
	conv_vec->data[0] = 1;
	conv_vec->data[1] = 2;
	conv_vec->data[2] = 3;
	preds_vec = mnew(1, 1);
	mmul(preds2, conv_vec, preds_vec);
	y_vec = mnew(1, 1);
	mmul(y_test, conv_vec, y_vec);

	/* Make sure the accuracy is greater than the no-skill rate (i.e. random chance).
	The no-skill rate is just the frequency of the most frequent class. This would be the accuracy
	of a classifier that only guessed the most frequent class. For the iris dataset,
	with an even frequency of 3 classes, it would be 0.33 */
	acc = lacc(preds_vec, y_vec);
	mu_assert("Error, accuracy for classification is less than no skill rate!", acc > 0.33);

	/* Free unneeded variables */
	mfree(preds_vec);
	mfree(y_vec);
	mfree(conv_vec);
	mfree(preds);
	mfree(preds2);
	mfree(iris);
	mfree(iris_train);
	mfree(iris_test);
	mfree(X_train);
	mfree(X_test);
	mfree(y_train);
	mfree(y_test);
	free(nn); // TODO: Should this be freed differently.

	return NULL;
}

static char* all_tests(){
	mu_run_test(test_mnew);
	mu_run_test(test_mcmp);
	mu_run_test(test_mmul);
	mu_run_test(test_madd);
	mu_run_test(test_mscale);
	mu_run_test(test_mhad);
	mu_run_test(test_mtrns);
	mu_run_test(test_mslice);
	mu_run_test(test_mfrob);
	mu_run_test(test_mrand);
	mu_run_test(test_asmax);
	mu_run_test(test_lmse);
	mu_run_test(test_ninit);
	mu_run_test(test_npred);
	mu_run_test(test_ndiff);
	mu_run_test(test_nbprop);
	mu_run_test(test_ntrain);
	mu_run_test(test_npredm);
	mu_run_test(test_io);
	mu_run_test(test_categorical);
	return NULL;
}

// TODO: Print number of successes out of total
int main(void){
	srand(42);
	char* result = all_tests();
	if(result != NULL){
		printf("%s\n",result);
		printf("Not all tests passed!\n");
	}
	else{
		printf("All tests passed!\n");
	}
	printf("Tests run: %d\n", tests_run);

	return result != NULL;
}