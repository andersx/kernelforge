#include <cblas.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>

void kernel_symm(
    const double* Xptr,
    int n,
    int rep_size,
    double alpha,
    double* Kptr
) {
    // 1) DSYRK (RowMajor, lower triangle): K = (-2*alpha) * X * X^T + 0*K
    double t0 = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size,
                -2.0 * alpha,
                Xptr, rep_size,
                0.0, Kptr, n);
    double t1 = omp_get_wtime();
    std::cout << "dsyrk took " << (t1 - t0) << " seconds\n";

    // 2) diag = -0.5 * diag(K)
    std::vector<double> diag(n);
    for (int i = 0; i < n; ++i) {
        diag[i] = -0.5 * Kptr[i * n + i];
    }

    // 3) K += 1 * (1*diag^T + diag*1^T) on LOWER via dsyr2
    std::vector<double> onevec(n, 1.0);
    cblas_dsyr2(CblasRowMajor, CblasLower, n, 1.0,
                onevec.data(), 1,
                diag.data(), 1,
                Kptr, n);

    // 4) Elementwise exp on the lower triangle (i <= j)
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            Kptr[j * n + i] = std::exp(Kptr[j * n + i]);
        }
    }
}

static inline void rowwise_self_norms(const double* X, std::size_t n, std::size_t d, double* out) {
    // out[i] = sum_k X[i, k]^2  (row-major: row i is contiguous)
    for (std::size_t i = 0; i < n; ++i) {
        const double* row = X + i * d;
        double acc = 0.0;
        for (std::size_t k = 0; k < d; ++k) acc += row[k] * row[k];
        out[i] = acc;
    }
}

void kernel_asymm(const double* X1, 
                  const double* X2,
                  std::size_t n1, 
                  std::size_t n2, 
                  std::size_t d,
                  double alpha,
                  double* K)
{
    // 1) K = (-2*alpha) * X2 * X1^T
    // RowMajor: A=X2 (n2 x d), B=X1 (n1 x d) but we pass Trans(B) => (d x n1)
    // lda = d, ldb = d, ldc = n1
    const double gemm_alpha = -2.0 * alpha;
    const double gemm_beta  = 0.0;
    double t0 = omp_get_wtime();
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                static_cast<int>(n1),       // M
                static_cast<int>(n2),       // N
                static_cast<int>(d),        // K
                gemm_alpha,
                X1, static_cast<int>(d),    // A, lda
                X2, static_cast<int>(d),    // B, ldb  (Trans)
                gemm_beta,
                K,  static_cast<int>(n2));  // C, ldc  (n2 x n1)
    double t1 = omp_get_wtime();
    std::cout << "dgemm took " << (t1 - t0) << " seconds\n";

    // 2) Rowwise self norms
    std::vector<double> nrm1(n1), nrm2(n2);
    rowwise_self_norms(X1, n1, d, nrm1.data());
    rowwise_self_norms(X2, n2, d, nrm2.data());

    // Ones
    std::vector<double> one_n1(n1, 1.0), one_n2(n2, 1.0);

    // 3) K += alpha * (ones_n2 * nrm1^T)   => GER(m=n2, n=n1)
    cblas_dger(CblasRowMajor,
               static_cast<int>(n1), static_cast<int>(n2),
               alpha,
               one_n1.data(), 1,
               nrm2.data(),   1,
               K, static_cast<int>(n2));

    // 4) K += alpha * (nrm2 * ones_n1^T)
    cblas_dger(CblasRowMajor,
               static_cast<int>(n1), static_cast<int>(n2),
               alpha,
               nrm1.data(),   1,
               one_n2.data(), 1,
               K, static_cast<int>(n2));

    // 5) Elementwise exp
    const std::size_t total = n2 * n1;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(total); ++idx) {
        K[static_cast<std::size_t>(idx)] = std::exp(K[static_cast<std::size_t>(idx)]);
    }
}
