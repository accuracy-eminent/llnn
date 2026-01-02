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
	mu_assert("Values are wrong!", out->data[3] != 5 || out->data[0] != 1);
	// TODO: Test transposed
	mslice(a, out, 0, 2, 1);
	mu_assert("Sliced size is wrong!", out->rows == 2 && out->cols == 2);
	mu_assert("Values are wrong!", out->data[3] != 5 || out->data[0] != 1);

	/* Free variables */
	mfree(a);
	mfree(b);
	mfree(out);

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
