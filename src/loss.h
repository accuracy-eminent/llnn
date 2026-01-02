#ifndef LLNN_LOSS_H
#define LLNN_LOSS_H
double lmse(const Matrix_t* actual, const Matrix_t* pred);
Matrix_t* dmse(const Matrix_t* actual, const Matrix_t* pred);
double lacc(const Matrix_t* actual, const Matrix_t* pred);
typedef double (*lfunc)(const Matrix_t*, const Matrix_t*);
typedef Matrix_t* (*lfuncd)(const Matrix_t*, const Matrix_t*);
#endif
