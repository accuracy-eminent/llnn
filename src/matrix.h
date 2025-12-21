#ifndef LLNN_MATRIX_H
#define LLNN_MATRIX_H
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix_t;

#define GET_M(m, r, c) ((m).data[((r) * (m).cols) + (col)])
#define SET_M(m, r, c, val) (m).data[((r) * (m).cols) + (col)] = (val)
#define IDX_M(m, r, c) (((r) * (m).cols) + (col))

#endif