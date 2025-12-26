#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "minunit.h"
#include "../src/matrix.h"

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

static char* all_tests(){
	mu_run_test(test_mnew);
	mu_run_test(test_mcmp);
	mu_run_test(test_mmul);
	mu_run_test(test_madd);
	return NULL;
}


int main(void){
	char* result = all_tests();
	if(result != NULL){
		printf("%s\n",result);
	}
	else{
		printf("All tests passed!\n");
	}
	printf("Tests run: %d\n", tests_run);

	return result != NULL;
}
