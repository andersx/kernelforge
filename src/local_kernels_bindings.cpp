// C++ standard library
#include <limits>
#include <stdexcept>
#include <string>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "local_kernels.hpp"

namespace py = pybind11;

static inline void check_shape(const py::array &arr, std::initializer_list<py::ssize_t> want) {
    if (arr.ndim() != static_cast<py::ssize_t>(want.size()))
        throw std::invalid_argument("Array has wrong rank");
    size_t i = 0;
    for (auto w : want) {
        if (w >= 0 && arr.shape(i) != w)
            throw std::invalid_argument("Array has unexpected shape at dim " + std::to_string(i));
        ++i;
    }
}

static py::array_t<double> flocal_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,
    py::array_t<double, py::array::c_style | py::array::forcecast> x2,
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,
    py::array_t<int, py::array::c_style | py::array::forcecast> q2,
    py::array_t<int, py::array::c_style | py::array::forcecast> n1,
    py::array_t<int, py::array::c_style | py::array::forcecast> n2, double sigma) {
    if (x1.ndim() != 3 || x2.ndim() != 3 || q1.ndim() != 2 || q2.ndim() != 2 || n1.ndim() != 1 ||
        n2.ndim() != 1) {
        throw std::invalid_argument(
            "Expect shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), q1(max_atoms1,nm1), "
            "q2(max_atoms2,nm2), n1(nm1), n2(nm2).");
    }

    const int nm1 = static_cast<int>(x1.shape(0));
    const int nm2 = static_cast<int>(x2.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int max_atoms2 = static_cast<int>(x2.shape(1));
    const int rep_size = static_cast<int>(x1.shape(2));

    // Cross-check all shapes
    if (x2.shape(2) != rep_size)
        throw std::invalid_argument("x2 rep_size mismatch.");
    if (q1.shape(1) != max_atoms1 || q1.shape(0) != nm1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (q2.shape(1) != max_atoms2 || q2.shape(0) != nm2)
        throw std::invalid_argument("q2 shape mismatch.");
    if (n1.shape(0) != nm1)
        throw std::invalid_argument("n1 length mismatch.");
    if (n2.shape(0) != nm2)
        throw std::invalid_argument("n2 length mismatch.");

    // Flatten into std::vector
    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t q1N = (size_t)max_atoms1 * nm1;
    const size_t q2N = (size_t)max_atoms2 * nm2;

    std::vector<double> x1v(x1N), x2v(x2N);
    std::vector<int> q1v(q1N), q2v(q2N), n1v(nm1), n2v(nm2);

    // Memory is C-contiguous because of c_style; we can memcpy via iterators
    std::copy_n(x1.data(), x1N, x1v.begin());
    std::copy_n(x2.data(), x2N, x2v.begin());
    std::copy_n(q1.data(), q1N, q1v.begin());
    std::copy_n(q2.data(), q2N, q2v.begin());
    std::copy_n(n1.data(), (size_t)nm1, n1v.begin());
    std::copy_n(n2.data(), (size_t)nm2, n2v.begin());

    // Make aligned (n2 x n1) output
    const std::size_t nelems = nm2 * nm1;
    double *Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        {static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(nm2)},
        {static_cast<py::ssize_t>(nm2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        Kptr, capsule);

        kf::fchl19::kernel_gaussian(x1v, x2v, q1v, q2v, n1v, n2v, nm1, nm2, max_atoms1, max_atoms2, rep_size,
                                    sigma, Kptr);

    return K;
}

static py::array_t<double> flocal_kernel_symm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,
    py::array_t<int, py::array::c_style | py::array::forcecast> n1, double sigma) {
    if (x1.ndim() != 3 || q1.ndim() != 2 || n1.ndim() != 1) {
        throw std::invalid_argument(
            "Expect shapes: x1(nm1,max_atoms1,rep), q1(max_atoms1,nm1), n1(nm1),.");
    }

    const int nm1 = static_cast<int>(x1.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int rep_size = static_cast<int>(x1.shape(2));

    // Cross-check all shapes
    if (x1.shape(2) != rep_size)
        throw std::invalid_argument("x1 rep_size mismatch.");
    if (q1.shape(1) != max_atoms1 || q1.shape(0) != nm1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (n1.shape(0) != nm1)
        throw std::invalid_argument("n1 length mismatch.");

    // Flatten into std::vector
    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t q1N = (size_t)max_atoms1 * nm1;

    std::vector<double> x1v(x1N);
    std::vector<int> q1v(q1N), n1v(nm1);

    // Memory is C-contiguous because of c_style; we can memcpy via iterators
    std::copy_n(x1.data(), x1N, x1v.begin());
    std::copy_n(q1.data(), q1N, q1v.begin());
    std::copy_n(n1.data(), (size_t)nm1, n1v.begin());

    // Make aligned (n2 x n1) output
    const std::size_t nelems = nm1 * nm1;
    double *Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        {static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(nm1)},
        {static_cast<py::ssize_t>(nm1 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        Kptr, capsule);

    kf::fchl19::kernel_gaussian_symm(x1v, q1v, n1v, nm1, max_atoms1, rep_size, sigma, Kptr);

    return K;
}

// Python wrapper
static py::array_t<double> fatomic_local_gradient_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,  // (nm1, max_atoms1, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> x2,  // (nm2, max_atoms2, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast>
        dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,  // (nm1, max_atoms1)
    py::array_t<int, py::array::c_style | py::array::forcecast> q2,  // (nm2, max_atoms2)
    py::array_t<int, py::array::c_style | py::array::forcecast> n1,  // (nm1,)
    py::array_t<int, py::array::c_style | py::array::forcecast> n2,  // (nm2,)
    double sigma) {
    // --- shape checks ---
    if (x1.ndim() != 3 || x2.ndim() != 3 || dX2.ndim() != 4 || q1.ndim() != 2 || q2.ndim() != 2 ||
        n1.ndim() != 1 || n2.ndim() != 1) {
        throw std::invalid_argument(
            "Expected shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), "
            "dX2(nm2,max_atoms2,rep,3*max_atoms2), q1(nm1,max_atoms1), q2(nm2,max_atoms2), "
            "n1(nm1), n2(nm2).");
    }

    const int nm1 = static_cast<int>(x1.shape(0));
    const int nm2 = static_cast<int>(x2.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int max_atoms2 = static_cast<int>(x2.shape(1));
    const int rep_size = static_cast<int>(x1.shape(2));

    if (x2.shape(1) != max_atoms2 || x2.shape(2) != rep_size)
        throw std::invalid_argument("x2 shape mismatch.");
    if (dX2.shape(0) != nm2 || dX2.shape(1) != max_atoms2 || dX2.shape(2) != rep_size ||
        dX2.shape(3) != 3 * max_atoms2)
        throw std::invalid_argument(
            "dX2 shape mismatch (must be (nm2,max_atoms2,rep,3*max_atoms2)).");
    if (q1.shape(0) != nm1 || q1.shape(1) != max_atoms1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (q2.shape(0) != nm2 || q2.shape(1) != max_atoms2)
        throw std::invalid_argument("q2 shape mismatch.");
    if (n1.shape(0) != nm1)
        throw std::invalid_argument("n1 length mismatch.");
    if (n2.shape(0) != nm2)
        throw std::invalid_argument("n2 length mismatch.");

    // Flatten into std::vector (C-contiguous thanks to c_style|forcecast)
    const std::size_t x1N = static_cast<std::size_t>(nm1) * max_atoms1 * rep_size;
    const std::size_t x2N = static_cast<std::size_t>(nm2) * max_atoms2 * rep_size;
    const std::size_t dXN = static_cast<std::size_t>(nm2) * max_atoms2 * rep_size *
                            (3 * static_cast<std::size_t>(max_atoms2));
    const std::size_t q1N = static_cast<std::size_t>(nm1) * max_atoms1;
    const std::size_t q2N = static_cast<std::size_t>(nm2) * max_atoms2;

    std::vector<double> x1v(x1N), x2v(x2N), dX2v(dXN);
    std::vector<int> q1v(q1N), q2v(q2N), n1v(nm1), n2v(nm2);

    std::copy_n(x1.data(), x1N, x1v.begin());
    std::copy_n(x2.data(), x2N, x2v.begin());
    std::copy_n(dX2.data(), dXN, dX2v.begin());
    std::copy_n(q1.data(), q1N, q1v.begin());
    std::copy_n(q2.data(), q2N, q2v.begin());
    std::copy_n(n1.data(), static_cast<std::size_t>(nm1), n1v.begin());
    std::copy_n(n2.data(), static_cast<std::size_t>(nm2), n2v.begin());

    // Compute naq2 = 3 * sum(n2)
    long long naq2_ll = 0;
    for (int b = 0; b < nm2; ++b) {
        int nb = n2v[b];
        if (nb < 0)
            nb = 0;
        if (nb > max_atoms2)
            nb = max_atoms2;
        naq2_ll += 3ll * nb;
    }
    if (naq2_ll <= 0) {
        // Return an empty (nm1,0) array if there are no derivatives
        std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(nm1), 0};
        std::vector<py::ssize_t> strides{py::ssize_t(0), py::ssize_t(sizeof(double))};
        return py::array_t<double>(shape, strides);
    }
    if (naq2_ll > std::numeric_limits<py::ssize_t>::max())
        throw std::overflow_error("naq2 is too large.");
    const int naq2 = static_cast<int>(naq2_ll);

    // Create aligned output (nm1, naq2), row-major (a*naq2 + q)
    double *Kptr = aligned_alloc_64(static_cast<std::size_t>(nm1) * naq2);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        {static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(naq2)},
        {static_cast<py::ssize_t>(naq2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        Kptr, capsule);

    // Call C++ implementation (releases GIL for BLAS/OpenMP)
    {
        py::gil_scoped_release release;
        kf::fchl19::kernel_gaussian_jacobian(x1v, x2v, dX2v, q1v, q2v, n1v, n2v, nm1, nm2,
                                             max_atoms1, max_atoms2, rep_size, naq2, sigma, Kptr);
    }

    return K;
}

static py::array_t<double> fgdml_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,  // (nm1, max_atoms1, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> x2,  // (nm2, max_atoms2, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast>
        dx1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    py::array_t<double, py::array::c_style | py::array::forcecast>
        dx2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,  // (nm1, max_atoms1)
    py::array_t<int, py::array::c_style | py::array::forcecast> q2,  // (nm2, max_atoms2)
    py::array_t<int, py::array::c_style | py::array::forcecast> n1,  // (nm1,)
    py::array_t<int, py::array::c_style | py::array::forcecast> n2,  // (nm2,)
    double sigma) {
    // ---- shape checks ----
    if (x1.ndim() != 3 || x2.ndim() != 3 || dx1.ndim() != 4 || dx2.ndim() != 4 || q1.ndim() != 2 ||
        q2.ndim() != 2 || n1.ndim() != 1 || n2.ndim() != 1) {
        throw std::invalid_argument(
            "Expected: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), "
            "dx1(nm1,max_atoms1,rep,3*max_atoms1), dx2(nm2,max_atoms2,rep,3*max_atoms2), "
            "q1(nm1,max_atoms1), q2(nm2,max_atoms2), n1(nm1), n2(nm2).");
    }

    const int nm1 = static_cast<int>(x1.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int rep = static_cast<int>(x1.shape(2));

    const int nm2 = static_cast<int>(x2.shape(0));
    const int max_atoms2 = static_cast<int>(x2.shape(1));

    if (x2.shape(2) != rep)
        throw std::invalid_argument("x2 rep mismatch.");
    if (dx1.shape(0) != nm1 || dx1.shape(1) != max_atoms1 || dx1.shape(2) != rep ||
        dx1.shape(3) != 3 * max_atoms1)
        throw std::invalid_argument("dx1 shape mismatch.");
    if (dx2.shape(0) != nm2 || dx2.shape(1) != max_atoms2 || dx2.shape(2) != rep ||
        dx2.shape(3) != 3 * max_atoms2)
        throw std::invalid_argument("dx2 shape mismatch.");
    if (q1.shape(0) != nm1 || q1.shape(1) != max_atoms1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (q2.shape(0) != nm2 || q2.shape(1) != max_atoms2)
        throw std::invalid_argument("q2 shape mismatch.");
    if (n1.shape(0) != nm1)
        throw std::invalid_argument("n1 length mismatch.");
    if (n2.shape(0) != nm2)
        throw std::invalid_argument("n2 length mismatch.");

    // ---- flatten to std::vector (C-contiguous guaranteed by c_style|forcecast) ----
    const std::size_t x1N = (std::size_t)nm1 * max_atoms1 * rep;
    const std::size_t x2N = (std::size_t)nm2 * max_atoms2 * rep;
    const std::size_t dx1N = (std::size_t)nm1 * max_atoms1 * rep * (3 * (std::size_t)max_atoms1);
    const std::size_t dx2N = (std::size_t)nm2 * max_atoms2 * rep * (3 * (std::size_t)max_atoms2);
    const std::size_t q1N = (std::size_t)nm1 * max_atoms1;
    const std::size_t q2N = (std::size_t)nm2 * max_atoms2;

    std::vector<double> x1v(x1N), x2v(x2N), dx1v(dx1N), dx2v(dx2N);
    std::vector<int> q1v(q1N), q2v(q2N), n1v(nm1), n2v(nm2);

    std::copy_n(x1.data(), x1N, x1v.begin());
    std::copy_n(x2.data(), x2N, x2v.begin());
    std::copy_n(dx1.data(), dx1N, dx1v.begin());
    std::copy_n(dx2.data(), dx2N, dx2v.begin());
    std::copy_n(q1.data(), q1N, q1v.begin());
    std::copy_n(q2.data(), q2N, q2v.begin());
    std::copy_n(n1.data(), (std::size_t)nm1, n1v.begin());
    std::copy_n(n2.data(), (std::size_t)nm2, n2v.begin());

    // ---- compute naq1 = 3*sum_a min(max(n1[a],0), max_atoms1), naq2 analogously ----
    long long naq1_ll = 0, naq2_ll = 0;
    for (int a = 0; a < nm1; ++a) {
        int na = n1v[a];
        if (na < 0)
            na = 0;
        if (na > max_atoms1)
            na = max_atoms1;
        naq1_ll += 3ll * na;
    }
    for (int b = 0; b < nm2; ++b) {
        int nb = n2v[b];
        if (nb < 0)
            nb = 0;
        if (nb > max_atoms2)
            nb = max_atoms2;
        naq2_ll += 3ll * nb;
    }
    if (naq1_ll < 0 || naq2_ll < 0)
        throw std::overflow_error("naq1/naq2 negative?");
    if (naq1_ll > std::numeric_limits<py::ssize_t>::max() ||
        naq2_ll > std::numeric_limits<py::ssize_t>::max())
        throw std::overflow_error("naq1/naq2 too large.");
    const int naq1 = static_cast<int>(naq1_ll);
    const int naq2 = static_cast<int>(naq2_ll);

    // If any side has zero derivatives, return an empty shaped array
    if (naq1 == 0 || naq2 == 0) {
        py::array_t<double> K(py::array::ShapeContainer{(py::ssize_t)naq2, (py::ssize_t)naq1},
                              py::array::StridesContainer{(py::ssize_t)(naq1 * sizeof(double)),
                                                          (py::ssize_t)sizeof(double)});
        return K;  // NumPy owns a small empty buffer
    }

    // ---- aligned output allocation (naq2 x naq1), row-major ----
    double *Kptr = aligned_alloc_64((std::size_t)naq2 * naq1);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        /*shape*/ py::array::ShapeContainer{(py::ssize_t)naq2, (py::ssize_t)naq1},
        /*strides*/
        py::array::StridesContainer{(py::ssize_t)(naq1 * sizeof(double)),
                                    (py::ssize_t)sizeof(double)},
        /*ptr*/ Kptr,
        /*base*/ capsule);

    // ---- call C++ (release GIL) ----
    {
        py::gil_scoped_release release;
        kf::fchl19::kernel_gaussian_hessian(x1v, x2v, dx1v, dx2v, q1v, q2v, n1v, n2v, nm1, nm2, max_atoms1,
                                            max_atoms2, rep, naq1, naq2, sigma, Kptr);
    }

    return K;
}

static py::array_t<double> fgdml_kernel_symm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,  // (nm1, max_atoms1, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast>
        dx1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,  // (nm1, max_atoms1)
    py::array_t<int, py::array::c_style | py::array::forcecast> n1,  // (nm1,)
    double sigma) {
    // ---- shape checks ----
    if (x1.ndim() != 3 || dx1.ndim() != 4 || q1.ndim() != 2 || n1.ndim() != 1) {
        throw std::invalid_argument("Expected: x1(nm1,max_atoms1,rep),  "
                                    "dx1(nm1,max_atoms1,rep,3*max_atoms1), "
                                    "q1(nm1,max_atoms1), n1(nm1).");
    }

    const int nm1 = static_cast<int>(x1.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int rep = static_cast<int>(x1.shape(2));

    if (dx1.shape(0) != nm1 || dx1.shape(1) != max_atoms1 || dx1.shape(2) != rep ||
        dx1.shape(3) != 3 * max_atoms1)
        throw std::invalid_argument("dx1 shape mismatch.");
    if (q1.shape(0) != nm1 || q1.shape(1) != max_atoms1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (n1.shape(0) != nm1)
        throw std::invalid_argument("n1 length mismatch.");

    // ---- flatten to std::vector (C-contiguous guaranteed by c_style|forcecast) ----
    const std::size_t x1N = (std::size_t)nm1 * max_atoms1 * rep;
    const std::size_t dx1N = (std::size_t)nm1 * max_atoms1 * rep * (3 * (std::size_t)max_atoms1);
    const std::size_t q1N = (std::size_t)nm1 * max_atoms1;

    std::vector<double> x1v(x1N), dx1v(dx1N);
    std::vector<int> q1v(q1N), n1v(nm1);

    std::copy_n(x1.data(), x1N, x1v.begin());
    std::copy_n(dx1.data(), dx1N, dx1v.begin());
    std::copy_n(q1.data(), q1N, q1v.begin());
    std::copy_n(n1.data(), (std::size_t)nm1, n1v.begin());

    // ---- compute naq1 = 3*sum_a min(max(n1[a],0), max_atoms1), naq2 analogously ----
    long long naq1_ll = 0;
    for (int a = 0; a < nm1; ++a) {
        int na = n1v[a];
        if (na < 0)
            na = 0;
        if (na > max_atoms1)
            na = max_atoms1;
        naq1_ll += 3ll * na;
    }
    if (naq1_ll < 0)
        throw std::overflow_error("naq1/naq2 negative?");
    if (naq1_ll > std::numeric_limits<py::ssize_t>::max())
        throw std::overflow_error("naq1/naq2 too large.");
    const int naq1 = static_cast<int>(naq1_ll);

    // If any side has zero derivatives, return an empty shaped array
    if (naq1 == 0) {
        py::array_t<double> K(py::array::ShapeContainer{(py::ssize_t)naq1, (py::ssize_t)naq1},
                              py::array::StridesContainer{(py::ssize_t)(naq1 * sizeof(double)),
                                                          (py::ssize_t)sizeof(double)});
        return K;  // NumPy owns a small empty buffer
    }

    // ---- aligned output allocation (naq2 x naq1), row-major ----
    double *Kptr = aligned_alloc_64((std::size_t)naq1 * naq1);
    auto capsule = py::capsule(Kptr, [](void *p) { aligned_free_64(p); });

    py::array_t<double> K(
        /*shape*/ py::array::ShapeContainer{(py::ssize_t)naq1, (py::ssize_t)naq1},
        /*strides*/
        py::array::StridesContainer{(py::ssize_t)(naq1 * sizeof(double)),
                                    (py::ssize_t)sizeof(double)},
        /*ptr*/ Kptr,
        /*base*/ capsule);

    // ---- call C++ (release GIL) ----
    {
        py::gil_scoped_release release;
        kf::fchl19::kernel_gaussian_hessian_symm(x1v, dx1v, q1v, n1v, nm1, max_atoms1, rep, naq1, sigma,
                                                 Kptr);
    }

    return K;
}

// RFP-output (TRANSR='N', UPLO='U'): returns a 1-D array of length nm*(nm+1)/2
static py::array_t<double> flocal_kernel_symm_rfp_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,
    py::array_t<int, py::array::c_style | py::array::forcecast> q1,
    py::array_t<int, py::array::c_style | py::array::forcecast> n1, double sigma) {
    // Expect shapes: x1(nm, max_atoms, rep), q1(nm, max_atoms), n1(nm)
    if (x1.ndim() != 3 || q1.ndim() != 2 || n1.ndim() != 1) {
        throw std::invalid_argument("Expect x1(nm,max_atoms,rep), q1(nm,max_atoms), n1(nm).");
    }

    const int nm = static_cast<int>(x1.shape(0));
    const int max_atoms = static_cast<int>(x1.shape(1));
    const int rep_size = static_cast<int>(x1.shape(2));

    // Cross-check shapes
    if (q1.shape(0) != nm || q1.shape(1) != max_atoms)
        throw std::invalid_argument("q1 shape mismatch.");
    if (n1.shape(0) != nm)
        throw std::invalid_argument("n1 length mismatch.");

    // Flatten into std::vector
    const size_t xN = (size_t)nm * (size_t)max_atoms * (size_t)rep_size;
    const size_t qN = (size_t)nm * (size_t)max_atoms;

    std::vector<double> xv(xN);
    std::vector<int> qv(qN), nv(nm);

    std::copy_n(x1.data(), xN, xv.begin());
    std::copy_n(q1.data(), qN, qv.begin());
    std::copy_n(n1.data(), (size_t)nm, nv.begin());

    // Allocate aligned RFP output: nt = nm*(nm+1)/2
    const size_t nt = (size_t)nm * (nm + 1ull) / 2ull;
    double *arf_ptr = aligned_alloc_64(nt);
    if (!arf_ptr)
        throw std::bad_alloc();

    auto capsule = py::capsule(arf_ptr, [](void *p) { aligned_free_64(p); });

    // Call kernel (fills arf_ptr in TRANSR='N', UPLO='U')
    kf::fchl19::kernel_gaussian_symm_rfp(xv, qv, nv, nm, max_atoms, rep_size, sigma, arf_ptr);

    // Return as 1-D C-contiguous array without extra copies
    return py::array_t<double>({(py::ssize_t)nt}, {(py::ssize_t)sizeof(double)}, arf_ptr, capsule);
}

PYBIND11_MODULE(local_kernels, m) {
    m.doc() = "Local (atom-pair-wise) kernels for FCHL19";

    m.def("kernel_gaussian_hessian_symm", &fgdml_kernel_symm_py, py::arg("x1"), py::arg("dx1"), py::arg("q1"),
          py::arg("n1"), py::arg("sigma"),
          R"(Compute the Gaussian GDML/Hessian kernel (symmetric).

Args:
  x1:  (nm1, max_atoms1, rep)
  dx1: (nm1, max_atoms1, rep, 3*max_atoms1)
  q1:  (nm1, max_atoms1)
  n1:  (nm1,)
  sigma: positive float

Returns:
  K: (naq1, naq1) row-major, where naq1 = 3 * sum(n1)
)");

    m.def("kernel_gaussian_hessian", &fgdml_kernel_py, py::arg("x1"), py::arg("x2"), py::arg("dx1"),
          py::arg("dx2"), py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
          py::arg("sigma"),
          R"(Compute the Gaussian GDML/Hessian kernel.

Args:
  x1:  (nm1, max_atoms1, rep)
  x2:  (nm2, max_atoms2, rep)
  dx1: (nm1, max_atoms1, rep, 3*max_atoms1)
  dx2: (nm2, max_atoms2, rep, 3*max_atoms2)
  q1:  (nm1, max_atoms1)
  q2:  (nm2, max_atoms2)
  n1:  (nm1,)
  n2:  (nm2,)
  sigma: positive float

Returns:
  K: (naq2, naq1) row-major, where
     naq1 = 3 * sum(n1), naq2 = 3 * sum(n2)
)");

    m.def("kernel_gaussian_jacobian", &fatomic_local_gradient_kernel_py, py::arg("x1"),
          py::arg("x2"), py::arg("dX2"), py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
          py::arg("sigma"),
          R"(Compute the Gaussian Jacobian kernel (gradient w.r.t. coordinates of set-2).

Args:
  x1:  (nm1, max_atoms1, rep)
  x2:  (nm2, max_atoms2, rep)
  dX2: (nm2, max_atoms2, rep, 3*max_atoms2)
  q1:  (nm1, max_atoms1)
  q2:  (nm2, max_atoms2)
  n1:  (nm1,)
  n2:  (nm2,)
  sigma: positive float

Returns:
  K: (nm1, naq2) where naq2 = 3 * sum(n2)
)");

    m.def("kernel_gaussian", &flocal_kernel_py, py::arg("x1"), py::arg("x2"), py::arg("q1"),
          py::arg("q2"), py::arg("n1"), py::arg("n2"), py::arg("sigma"));
    m.def("kernel_gaussian_symm", &flocal_kernel_symm_py, py::arg("x1"), py::arg("q1"), py::arg("n1"),
          py::arg("sigma"));
    m.def("kernel_gaussian_symm_rfp", &flocal_kernel_symm_rfp_py, py::arg("x1"), py::arg("q1"),
          py::arg("n1"), py::arg("sigma"),
          "Symmetric Gaussian kernel with RFP (TRANSR='N', UPLO='U') output.\n"
          "Inputs: x1(nm,max_atoms,rep), q1(nm,max_atoms), n1(nm).\n"
          "Returns: 1-D RFP array of length nm*(nm+1)/2.");
}
