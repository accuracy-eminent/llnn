#ifndef LLNN_ACTIV_H
#define LLNN_ACTIV_H
double arelu(double x);
double drelu(double output);
double alrelu(double x);
double alin(double x);
double asigm(double x);
double dsigm(double x);
double atanh(double x);
Matrix_t* asmax(const Matrix_t* a);
Matrix_t* ahmax(const Matrix_t* a);
#endif
