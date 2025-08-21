#ifndef KERNELFORGE_H
#define KERNELFORGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI for the Fortran routine */
int compute_inverse_distance(const double* x_3_by_n, int n, double* d_packed);

#ifdef __cplusplus
}
#endif

#endif

