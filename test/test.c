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

static char* all_tests(){
	mu_run_test(test_mnew);
	mu_run_test(test_mcmp);
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
