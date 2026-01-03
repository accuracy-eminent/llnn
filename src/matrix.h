#ifndef LLNN_MATRIX_H
#define LLNN_MATRIX_H
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix_t;

#define GET_M(m, r, c) ((m).data[((r) * (m).cols) + (c)])
#define SET_M(m, r, c, val) (m).data[((r) * (m).cols) + (c)] = (val)
#define IDX_M(m, r, c) (((r) * (m).cols) + (c))

typedef double (*dfunc)(double);
typedef Matrix_t* (*mfunc)(const Matrix_t*);

int mrealloc(Matrix_t *x, int rows, int cols);
Matrix_t* mnew(int rows, int cols);
void mfree(Matrix_t* x);
Matrix_t* mmul(const Matrix_t* a, const Matrix_t* b, Matrix_t *out);
int mcmp(const Matrix_t* a, const Matrix_t* b);
Matrix_t* madd(const Matrix_t* a, const Matrix_t* b, Matrix_t* out);
Matrix_t* msub(const Matrix_t* a, const Matrix_t* b, Matrix_t* out);
double mfrob(const Matrix_t* a);
Matrix_t* mscale(const Matrix_t* a, double b, Matrix_t* out);
Matrix_t* mrand(int rows, int cols, double min, double max, Matrix_t* out);
Matrix_t* mhad(const Matrix_t* a, const Matrix_t* b, Matrix_t* out);
Matrix_t* mtrns(const Matrix_t* a, Matrix_t* out);
Matrix_t* mslice(const Matrix_t* in, Matrix_t* out, int start, int stop, int t);
Matrix_t* mapply(const Matrix_t* a, dfunc func, Matrix_t* out);
void mprint(const Matrix_t* x);

#endif