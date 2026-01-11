#include "matrix.h"
#include "nn.h"
#include "loss.h"
#include "llnn.h"


double lmse(const Matrix_t* actual, const Matrix_t* pred){
	double sum_square_error = 0, mean_square_error = 0;

	if(actual->rows != pred->rows){
		return 0.0;
	}

	for(int i = 0; i < actual->rows * actual->cols; i++){
		sum_square_error += SQR((actual->data[i] - pred->data[i]));
	}
	mean_square_error = sum_square_error / (actual->rows * actual->cols);;

	return mean_square_error;
}

// Derivative of MSE
Matrix_t* dmse(const Matrix_t* actual, const Matrix_t* pred){
    // TODO: Make output matrix a parameter?
    Matrix_t *out = mnew(actual->rows, actual->cols);
	msub(actual, pred, out);
	return out;
}

double lacc(const Matrix_t* actual, const Matrix_t* pred){
	int row, col, successes, failures;
	double acc;

	// Validate inputs
	if((actual->rows != pred->rows) || (actual->cols != pred->cols)){
        // TODO: Error message
		return 0.0f;
	}

	// Calculate accuracy
	// First get the number of successes (matches) and non-matches (failures)
	successes = 0;
	failures = 0;
	for(int i = 0; i < actual->rows * actual->cols; i++){
        if(actual->data[i] == pred->data[i]) successes += 1;
        else failures += 1;
	}
	// Then divide the number of successes by the total
	acc = (double)(successes) / ( (double)(successes) + (double)(failures) );

	return acc;
}