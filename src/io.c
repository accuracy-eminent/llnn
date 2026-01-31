#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "llnn.h"
#include "matrix.h"

Matrix_t* ireadcsv(char *filename, Matrix_t *out){
	FILE *file;
	char row[4096];
	char* token;
	char ch;
	int row_num = 0;
	int col_num = 0;
	int rows = 0;
	int cols = 0;

	file = fopen(filename, "r");
    if(file == NULL)
    {
        fprintf(stderr, "Could not open file!\n");
        return NULL;
    }
	// Count the number of lines
	while(!feof(file)){
		ch = fgetc(file);
		if(ch == '\n')rows++;
	}
	rows -= 1; //Make sure to not count the header row
	fseek(file, 0, SEEK_SET);

    if(out == NULL) return NULL;

	// Loop through the rows of the csv
	while(fgets(row, 4096, file) != NULL){
		cols = 0;
		// For the first row (the header), get the number of columns
		if(row_num == 0){
			token = strtok(row, ",");
			while(token != NULL){
				token = strtok(NULL, ",");
				cols++;
			}
            // Allocate with calculated rows and cols
            if(mrealloc(out, rows, cols) != 0) return NULL;
		}
		// For every other row, fill in the data for each column
		else {
			col_num = 0;
			token = strtok(row, ",");
			out->data[IDX_M(*out, row_num - 1, col_num)] = strtod(token, NULL);
			col_num = 1;
			while(token != NULL){
				token = strtok(NULL, ",");
				if(token != NULL){
					out->data[IDX_M(*out, row_num - 1, col_num)] = strtod(token, NULL);
				}
				col_num++;
			}
		}
		row_num++;
	}

	// Clean up
	fclose(file);

	return out;
}
