#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "activ.h"
#include "loss.h"
#include "nn.h"
#include "io.h"

static char* test_categorical(){
	Matrix_t *iris, *iris_train, *iris_test, *X_train, *y_train, *X_test, *y_test;
	Matrix_t *preds, *preds2, *preds_vec, *y_vec, *conv_vec;
	llnn_network_t *nn;
	double acc;

	/* Load in data, split into train and test */
	iris = mnew(1, 1);
	ireadcsv("datasets/iris2.csv", iris);
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

int main(void)
{
    test_categorical();
    return 0;
}