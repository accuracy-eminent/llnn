#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"

int tests_run = 0;

static char* all_tests(){;
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
