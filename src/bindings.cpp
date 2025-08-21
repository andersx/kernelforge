#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
extern "C" {
  int compute_inverse_distance(const double* x_3_by_n, int n, double* d_packed);
}

namespace py = pybind11;

py::array_t<double> inverse_distance(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    auto buf = X.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("X must have shape (N,3)");
    }
    const int n = static_cast<int>(buf.shape[0]);

    // D packed length
    const ssize_t m = static_cast<ssize_t>(n) * (n - 1) / 2;
    auto D = py::array_t<double>(m);

    // Pass row-major (N,3) as transposed view (3,N) to Fortran without copy:
    // NumPy will give a view; pybind11 exposes data pointer for the view.
    py::array_t<double> XT({3, n}, {buf.strides[1], buf.strides[0]}, static_cast<double*>(buf.ptr), X);

    int rc = compute_inverse_distance(static_cast<const double*>(XT.request().ptr), n,
                                      static_cast<double*>(D.request().ptr));
    if (rc != 0) throw std::runtime_error("compute_inverse_distance failed with code " + std::to_string(rc));
    return D;
}

PYBIND11_MODULE(kernelforge, m) {
    m.doc() = "kernelforge: Fortran kernels with C ABI and Python bindings";
    m.def("inverse_distance", &inverse_distance, "Compute packed inverse distance matrix from (N,3) coordinates");
}

