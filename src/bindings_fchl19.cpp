#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <limits>
#include "aligned_alloc64.hpp"

#include "fchl19_representation.hpp"

namespace py = pybind11;

// Convert a 2D NumPy array (n,3) or 1D (n*3,) to std::vector<double>
static std::vector<double> as_coords_vector(const py::array& arr, size_t natoms) {
    if (arr.ndim() == 2) {
        if (static_cast<size_t>(arr.shape(0)) != natoms || arr.shape(1) != 3)
            throw std::invalid_argument("coords must have shape (n_atoms, 3)");
        std::vector<double> out(natoms * 3);
        auto buf = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = buf.unchecked<2>();
        for (size_t i = 0; i < natoms; ++i) {
            out[3*i+0] = r(i,0);
            out[3*i+1] = r(i,1);
            out[3*i+2] = r(i,2);
        }
        return out;
    } else if (arr.ndim() == 1) {
        if (static_cast<size_t>(arr.shape(0)) != natoms * 3)
            throw std::invalid_argument("coords 1D length must be n_atoms*3");
        auto buf = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = buf.unchecked<1>();
        return std::vector<double>(r.data(0), r.data(0) + natoms * 3);
    } else {
        throw std::invalid_argument("coords must be a 1D or 2D NumPy array");
    }
}

static std::vector<int> as_int_vector(const py::array& arr, size_t expected = SIZE_MAX) {
    auto buf = arr.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    if (buf.ndim() != 1) throw std::invalid_argument("array must be 1D");
    if (expected != SIZE_MAX && static_cast<size_t>(buf.shape(0)) != expected)
        throw std::invalid_argument("unexpected array length");
    auto r = buf.unchecked<1>();
    return std::vector<int>(r.data(0), r.data(0) + r.shape(0));
}

static std::vector<double> as_double_vector(const py::array& arr) {
    auto buf = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (buf.ndim() != 1) throw std::invalid_argument("array must be 1D");
    auto r = buf.unchecked<1>();
    return std::vector<double>(r.data(0), r.data(0) + r.shape(0));
}



py::array_t<double> generate_fchl_acsf_py(
    const py::array& coords,              // (n,3) or (3n,)
    const py::array& nuclear_z,           // (n,)
    std::vector<int> elements,
    int nRs2,
    int nRs3,
    int nFourier,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    const size_t natoms = static_cast<size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0) throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);

    // Default elements if not provided
    if (elements.empty()) {
        elements = {1,6,7,8,16};
    }

    // Construct Rs2, Rs3, Ts as in Python version
    std::vector<double> Rs2_v, Rs3_v, Ts_v;
    Rs2_v.reserve(nRs2);
    Rs3_v.reserve(nRs3);

    for (int i=1; i<=nRs2; ++i) {
        Rs2_v.push_back(rcut * static_cast<double>(i)/static_cast<double>(nRs2+1));
    }
    for (int i=1; i<=nRs3; ++i) {
        Rs3_v.push_back(acut * static_cast<double>(i)/static_cast<double>(nRs3+1));
    }
    Ts_v.reserve(2*nFourier);
    for (int i=0; i<2*nFourier; ++i) {
        Ts_v.push_back(M_PI * static_cast<double>(i)/static_cast<double>(2*nFourier-1));
    }

    // Normalization constant for three-body
    double norm_three_body_weight = std::sqrt(eta3/ M_PI) * three_body_weight;

    const size_t rep_size = fchl19::compute_rep_size(elements.size(), Rs2_v.size(), Rs3_v.size(), Ts_v.size());

    py::array_t<double> out({static_cast<py::ssize_t>(natoms), static_cast<py::ssize_t>(rep_size)});
    auto out_mut = out.mutable_unchecked<2>();

    std::vector<double> rep;
    {
        py::gil_scoped_release release;
        fchl19::generate_fchl_acsf(coords_v, z_v, elements, Rs2_v, Rs3_v, Ts_v,
                                   eta2, eta3, zeta, rcut, acut,
                                   two_body_decay, three_body_decay, norm_three_body_weight,
                                   rep);
    }

    if (rep.size() != natoms * rep_size) throw std::runtime_error("internal error: unexpected rep size");
    const double* src = rep.data();
    for (size_t i = 0; i < natoms; ++i)
        for (size_t j = 0; j < rep_size; ++j)
            out_mut(i, j) = src[i*rep_size + j];

    return out;
}


// Helper: flatten list of (n_atoms × d) arrays and (n_atoms,) charges into
// contiguous buffers and offsets.
static void flatten_inputs(const py::list& X_list,
                           const py::list& Q_list,
                           std::vector<double>& X,
                           std::vector<int>&    Q,
                           std::vector<std::size_t>& offsets,
                           std::size_t& d_out)
{
    const std::size_t nm = X_list.size();
    offsets.resize(nm+1); offsets[0]=0; d_out = 0;

    // First pass: shapes + total size
    std::size_t total = 0; bool d_set=false;
    for (std::size_t m=0; m<nm; ++m) {
        py::array x = py::cast<py::array>(X_list[m]);
        py::array q = py::cast<py::array>(Q_list[m]);
        if (x.ndim()!=2) throw std::runtime_error("Each rep must be 2D: (n_atoms × d)");
        if (q.ndim()!=1) throw std::runtime_error("Each charges array must be 1D");
        std::size_t n = static_cast<std::size_t>(x.shape(0));
        std::size_t d = static_cast<std::size_t>(x.shape(1));
        if (!d_set) { d_out = d; d_set=true; }
        else if (d_out!=d) throw std::runtime_error("All molecules must share the same rep dimension d");
        if (static_cast<std::size_t>(q.shape(0))!=n) throw std::runtime_error("charges length mismatch");
        total += n; offsets[m+1] = total;
    }

    X.resize(total * d_out);
    Q.resize(total);

    // Second pass: copy (row-major expected)
    std::size_t cursor = 0;
    for (std::size_t m=0; m<nm; ++m) {
        py::array x = py::cast<py::array>(X_list[m]);
        py::array q = py::cast<py::array>(Q_list[m]);
        x = py::array_t<double>(x); // enforce float64 view
        q = py::array_t<long long>(q); // enforce integer view
        const std::size_t n = static_cast<std::size_t>(x.shape(0));
        const std::size_t d = static_cast<std::size_t>(x.shape(1));
        const double* px = static_cast<const double*>(x.request().ptr);
        const long long* pq = static_cast<const long long*>(q.request().ptr);
        // copy reps
        std::copy(px, px + n*d, X.data() + cursor*d);
        // copy charges
        for (std::size_t i=0;i<n;++i) Q[cursor+i] = static_cast<int>(pq[i]);
        cursor += n;
    }
}


// Build basis arrays like in your Python: linspace(0,rcut,1+n)[1:]
static void build_basis_from_sizes(int nRs2, int nRs3, int nFourier,
                                   double rcut, double acut,
                                   std::vector<double>& Rs2,
                                   std::vector<double>& Rs3,
                                   std::vector<double>& Ts) {
    Rs2.clear(); Rs3.clear(); Ts.clear();
    Rs2.reserve(nRs2); Rs3.reserve(nRs3); Ts.reserve(2*nFourier);
    for (int i=1; i<=nRs2; ++i) Rs2.push_back(rcut * double(i)/double(nRs2+1));
    for (int i=1; i<=nRs3; ++i) Rs3.push_back(acut * double(i)/double(nRs3+1));
    const int nT = std::max(2*nFourier, 2); // ensure even >=2
    for (int i=0; i<nT; ++i) {
        double x = (nT==1 ? 0.0 : (double(i)/double(nT-1)));
        Ts.push_back(M_PI * x);
    }
}


static py::tuple generate_fchl_acsf_rep_and_grad_py(
    const py::array& coords,            // (n,3)
    const py::array& nuclear_z,         // (n,)
    std::vector<int> elements,
    int nRs2,
    int nRs3,
    int nFourier,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    const std::size_t natoms = static_cast<std::size_t>(nuclear_z.cast<py::array>().shape(0));
    if (natoms == 0) throw std::invalid_argument("n_atoms must be > 0");

    std::vector<double> coords_v = as_coords_vector(coords, natoms);
    std::vector<int> z_v = as_int_vector(nuclear_z, natoms);
    if (elements.empty()) elements = {1,6,7,8,16};

    std::vector<double> Rs2_v, Rs3_v, Ts_v;
    build_basis_from_sizes(nRs2, nRs3, nFourier, rcut, acut, Rs2_v, Rs3_v, Ts_v);

    // rescale three-body weight
    constexpr double PI = 3.14159265358979323846;
    const double w3 = std::sqrt(eta3 / PI) * three_body_weight;

    const std::size_t rep_size = fchl19::compute_rep_size(elements.size(), Rs2_v.size(), Rs3_v.size(), Ts_v.size());

    std::vector<double> rep, grad;
    {
        py::gil_scoped_release release;
        fchl19::generate_fchl_acsf_and_gradients(
            coords_v, z_v, elements, Rs2_v, Rs3_v, Ts_v,
            eta2, eta3, zeta, rcut, acut,
            two_body_decay, three_body_decay, w3,
            rep, grad);
    }

    // Wrap outputs
    py::array_t<double> rep_arr(py::array::ShapeContainer{ (py::ssize_t)natoms, (py::ssize_t)rep_size });
    auto R = rep_arr.mutable_unchecked<2>();
    for (std::size_t i=0;i<natoms;++i)
        for (std::size_t j=0;j<rep_size;++j)
            R(i,j) = rep[i*rep_size + j];

    // py::array_t<double> grad_arr(py::array::ShapeContainer{
    //     (py::ssize_t)natoms, (py::ssize_t)rep_size, (py::ssize_t)natoms, (py::ssize_t)3
    // });
    // auto G = grad_arr.mutable_unchecked<4>();
    // std::size_t idx = 0;
    // for (std::size_t i=0;i<natoms;++i)
    //     for (std::size_t j=0;j<rep_size;++j)
    //         for (std::size_t a=0;a<natoms;++a)
    //             for (int d=0; d<3; ++d)
    //                 G(i,j,a,d) = grad[idx++];
    
    py::array_t<double> grad_arr(py::array::ShapeContainer{
    (py::ssize_t)natoms, (py::ssize_t)rep_size, (py::ssize_t)(3*natoms)
        });
        auto G = grad_arr.mutable_unchecked<3>();
        std::size_t idx = 0;
        for (std::size_t i=0;i<natoms;++i)
          for (std::size_t j=0;j<rep_size;++j)
            for (std::size_t a=0;a<natoms;++a)
              for (int d=0; d<3; ++d)
                G(i, j, 3*a + d) = grad[idx++];



    return py::make_tuple(rep_arr, grad_arr);
}

static inline void check_shape(const py::array& arr, std::initializer_list<py::ssize_t> want) {
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
    py::array_t<int,    py::array::c_style | py::array::forcecast> q1,
    py::array_t<int,    py::array::c_style | py::array::forcecast> q2,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n1,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n2,
    double sigma) {

     if (x1.ndim() != 3 || x2.ndim() != 3 || q1.ndim() != 2 || q2.ndim() != 2 ||
         n1.ndim() != 1 || n2.ndim() != 1) {
         throw std::invalid_argument("Expect shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), q1(max_atoms1,nm1), q2(max_atoms2,nm2), n1(nm1), n2(nm2).");
     }

     const int nm1 = static_cast<int>(x1.shape(0));
     const int nm2 = static_cast<int>(x2.shape(0));
     const int max_atoms1 = static_cast<int>(x1.shape(1));
     const int max_atoms2 = static_cast<int>(x2.shape(1));
     const int rep_size   = static_cast<int>(x1.shape(2));

     // Cross-check all shapes
     if (x2.shape(2) != rep_size) throw std::invalid_argument("x2 rep_size mismatch.");
     if (q1.shape(1) != max_atoms1 || q1.shape(0) != nm1) throw std::invalid_argument("q1 shape mismatch.");
     if (q2.shape(1) != max_atoms2 || q2.shape(0) != nm2) throw std::invalid_argument("q2 shape mismatch.");
     if (n1.shape(0) != nm1) throw std::invalid_argument("n1 length mismatch.");
     if (n2.shape(0) != nm2) throw std::invalid_argument("n2 length mismatch.");

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
    double* Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    py::array_t<double> K(
        { static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(nm2) },
        { static_cast<py::ssize_t>(nm2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double)) },
        Kptr,
        capsule
    );

     fchl19::flocal_kernel(x1v, x2v, q1v, q2v, n1v, n2v,
                       nm1, nm2, max_atoms1, max_atoms2, rep_size, sigma, Kptr);

    return K;
}

static py::array_t<double> flocal_kernel_symm_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,
    py::array_t<int,    py::array::c_style | py::array::forcecast> q1,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n1,
    double sigma) {

     if (x1.ndim() != 3 || q1.ndim() != 2 || n1.ndim() != 1) {
         throw std::invalid_argument("Expect shapes: x1(nm1,max_atoms1,rep), q1(max_atoms1,nm1), n1(nm1),.");
     }

     const int nm1 = static_cast<int>(x1.shape(0));
     const int max_atoms1 = static_cast<int>(x1.shape(1));
     const int rep_size   = static_cast<int>(x1.shape(2));

     // Cross-check all shapes
     if (x1.shape(2) != rep_size) throw std::invalid_argument("x1 rep_size mismatch.");
     if (q1.shape(1) != max_atoms1 || q1.shape(0) != nm1) throw std::invalid_argument("q1 shape mismatch.");
     if (n1.shape(0) != nm1) throw std::invalid_argument("n1 length mismatch.");

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
    double* Kptr = aligned_alloc_64(nelems);
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    py::array_t<double> K(
        { static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(nm1) },
        { static_cast<py::ssize_t>(nm1 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double)) },
        Kptr,
        capsule
    );

    fchl19::flocal_kernel_symmetric(x1v, q1v,n1v, nm1, max_atoms1,rep_size, sigma, Kptr);

    return K;
}


// Python wrapper
static py::array_t<double> fatomic_local_gradient_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,   // (nm1, max_atoms1, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> x2,   // (nm2, max_atoms2, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    py::array_t<int,    py::array::c_style | py::array::forcecast> q1,   // (nm1, max_atoms1)
    py::array_t<int,    py::array::c_style | py::array::forcecast> q2,   // (nm2, max_atoms2)
    py::array_t<int,    py::array::c_style | py::array::forcecast> n1,   // (nm1,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> n2,   // (nm2,)
    double sigma
) {
    // --- shape checks ---
    if (x1.ndim() != 3 || x2.ndim() != 3 || dX2.ndim() != 4 ||
        q1.ndim() != 2 || q2.ndim() != 2 || n1.ndim() != 1 || n2.ndim() != 1) {
        throw std::invalid_argument(
            "Expected shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), "
            "dX2(nm2,max_atoms2,rep,3*max_atoms2), q1(nm1,max_atoms1), q2(nm2,max_atoms2), "
            "n1(nm1), n2(nm2).");
    }

    const int nm1        = static_cast<int>(x1.shape(0));
    const int nm2        = static_cast<int>(x2.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int max_atoms2 = static_cast<int>(x2.shape(1));
    const int rep_size   = static_cast<int>(x1.shape(2));

    if (x2.shape(1) != max_atoms2 || x2.shape(2) != rep_size)
        throw std::invalid_argument("x2 shape mismatch.");
    if (dX2.shape(0) != nm2 || dX2.shape(1) != max_atoms2 || dX2.shape(2) != rep_size
        || dX2.shape(3) != 3 * max_atoms2)
        throw std::invalid_argument("dX2 shape mismatch (must be (nm2,max_atoms2,rep,3*max_atoms2)).");
    if (q1.shape(0) != nm1 || q1.shape(1) != max_atoms1)
        throw std::invalid_argument("q1 shape mismatch.");
    if (q2.shape(0) != nm2 || q2.shape(1) != max_atoms2)
        throw std::invalid_argument("q2 shape mismatch.");
    if (n1.shape(0) != nm1) throw std::invalid_argument("n1 length mismatch.");
    if (n2.shape(0) != nm2) throw std::invalid_argument("n2 length mismatch.");

    // Flatten into std::vector (C-contiguous thanks to c_style|forcecast)
    const std::size_t x1N = static_cast<std::size_t>(nm1) * max_atoms1 * rep_size;
    const std::size_t x2N = static_cast<std::size_t>(nm2) * max_atoms2 * rep_size;
    const std::size_t dXN = static_cast<std::size_t>(nm2) * max_atoms2 * rep_size * (3 * static_cast<std::size_t>(max_atoms2));
    const std::size_t q1N = static_cast<std::size_t>(nm1) * max_atoms1;
    const std::size_t q2N = static_cast<std::size_t>(nm2) * max_atoms2;

    std::vector<double> x1v(x1N), x2v(x2N), dX2v(dXN);
    std::vector<int>    q1v(q1N), q2v(q2N), n1v(nm1), n2v(nm2);

    std::copy_n(x1.data(),  x1N, x1v.begin());
    std::copy_n(x2.data(),  x2N, x2v.begin());
    std::copy_n(dX2.data(), dXN, dX2v.begin());
    std::copy_n(q1.data(),  q1N, q1v.begin());
    std::copy_n(q2.data(),  q2N, q2v.begin());
    std::copy_n(n1.data(),  static_cast<std::size_t>(nm1), n1v.begin());
    std::copy_n(n2.data(),  static_cast<std::size_t>(nm2), n2v.begin());

    // Compute naq2 = 3 * sum(n2)
    long long naq2_ll = 0;
    for (int b = 0; b < nm2; ++b) {
        int nb = n2v[b];
        if (nb < 0) nb = 0;
        if (nb > max_atoms2) nb = max_atoms2;
        naq2_ll += 3ll * nb;
    }
    if (naq2_ll <= 0) {
        // Return an empty (nm1,0) array if there are no derivatives
        std::vector<py::ssize_t> shape{ static_cast<py::ssize_t>(nm1), 0 };
        std::vector<py::ssize_t> strides{ py::ssize_t(0), py::ssize_t(sizeof(double)) };
        return py::array_t<double>(shape, strides);
    }
    if (naq2_ll > std::numeric_limits<py::ssize_t>::max())
        throw std::overflow_error("naq2 is too large.");
    const int naq2 = static_cast<int>(naq2_ll);

    // Create aligned output (nm1, naq2), row-major (a*naq2 + q)
    double* Kptr = aligned_alloc_64(static_cast<std::size_t>(nm1) * naq2);
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    py::array_t<double> K(
        { static_cast<py::ssize_t>(nm1), static_cast<py::ssize_t>(naq2) },
        { static_cast<py::ssize_t>(naq2 * sizeof(double)), static_cast<py::ssize_t>(sizeof(double)) },
        Kptr,
        capsule
    );

    // Call C++ implementation (releases GIL for BLAS/OpenMP)
    {
        py::gil_scoped_release release;
        fchl19::fatomic_local_gradient_kernel(
            x1v, x2v, dX2v, q1v, q2v, n1v, n2v,
            nm1, nm2, max_atoms1, max_atoms2, rep_size, naq2, sigma,
            Kptr
        );
    }

    return K;
}

static py::array_t<double> fgdml_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> x1,   // (nm1, max_atoms1, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> x2,   // (nm2, max_atoms2, rep)
    py::array_t<double, py::array::c_style | py::array::forcecast> dx1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    py::array_t<double, py::array::c_style | py::array::forcecast> dx2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    py::array_t<int,    py::array::c_style | py::array::forcecast> q1,   // (nm1, max_atoms1)
    py::array_t<int,    py::array::c_style | py::array::forcecast> q2,   // (nm2, max_atoms2)
    py::array_t<int,    py::array::c_style | py::array::forcecast> n1,   // (nm1,)
    py::array_t<int,    py::array::c_style | py::array::forcecast> n2,   // (nm2,)
    double sigma
) {
    // ---- shape checks ----
    if (x1.ndim()!=3 || x2.ndim()!=3 || dx1.ndim()!=4 || dx2.ndim()!=4 ||
        q1.ndim()!=2 || q2.ndim()!=2 || n1.ndim()!=1 || n2.ndim()!=1) {
        throw std::invalid_argument(
            "Expected: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), "
            "dx1(nm1,max_atoms1,rep,3*max_atoms1), dx2(nm2,max_atoms2,rep,3*max_atoms2), "
            "q1(nm1,max_atoms1), q2(nm2,max_atoms2), n1(nm1), n2(nm2).");
    }

    const int nm1        = static_cast<int>(x1.shape(0));
    const int max_atoms1 = static_cast<int>(x1.shape(1));
    const int rep        = static_cast<int>(x1.shape(2));

    const int nm2        = static_cast<int>(x2.shape(0));
    const int max_atoms2 = static_cast<int>(x2.shape(1));

    if (x2.shape(2) != rep)                      throw std::invalid_argument("x2 rep mismatch.");
    if (dx1.shape(0)!=nm1 || dx1.shape(1)!=max_atoms1 || dx1.shape(2)!=rep || dx1.shape(3)!=3*max_atoms1)
        throw std::invalid_argument("dx1 shape mismatch.");
    if (dx2.shape(0)!=nm2 || dx2.shape(1)!=max_atoms2 || dx2.shape(2)!=rep || dx2.shape(3)!=3*max_atoms2)
        throw std::invalid_argument("dx2 shape mismatch.");
    if (q1.shape(0)!=nm1 || q1.shape(1)!=max_atoms1) throw std::invalid_argument("q1 shape mismatch.");
    if (q2.shape(0)!=nm2 || q2.shape(1)!=max_atoms2) throw std::invalid_argument("q2 shape mismatch.");
    if (n1.shape(0)!=nm1) throw std::invalid_argument("n1 length mismatch.");
    if (n2.shape(0)!=nm2) throw std::invalid_argument("n2 length mismatch.");

    // ---- flatten to std::vector (C-contiguous guaranteed by c_style|forcecast) ----
    const std::size_t x1N  = (std::size_t)nm1 * max_atoms1 * rep;
    const std::size_t x2N  = (std::size_t)nm2 * max_atoms2 * rep;
    const std::size_t dx1N = (std::size_t)nm1 * max_atoms1 * rep * (3 * (std::size_t)max_atoms1);
    const std::size_t dx2N = (std::size_t)nm2 * max_atoms2 * rep * (3 * (std::size_t)max_atoms2);
    const std::size_t q1N  = (std::size_t)nm1 * max_atoms1;
    const std::size_t q2N  = (std::size_t)nm2 * max_atoms2;

    std::vector<double> x1v(x1N), x2v(x2N), dx1v(dx1N), dx2v(dx2N);
    std::vector<int>    q1v(q1N), q2v(q2N), n1v(nm1), n2v(nm2);

    std::copy_n(x1.data(),  x1N,  x1v.begin());
    std::copy_n(x2.data(),  x2N,  x2v.begin());
    std::copy_n(dx1.data(), dx1N, dx1v.begin());
    std::copy_n(dx2.data(), dx2N, dx2v.begin());
    std::copy_n(q1.data(),  q1N,  q1v.begin());
    std::copy_n(q2.data(),  q2N,  q2v.begin());
    std::copy_n(n1.data(),  (std::size_t)nm1, n1v.begin());
    std::copy_n(n2.data(),  (std::size_t)nm2, n2v.begin());

    // ---- compute naq1 = 3*sum_a min(max(n1[a],0), max_atoms1), naq2 analogously ----
    long long naq1_ll = 0, naq2_ll = 0;
    for (int a = 0; a < nm1; ++a) {
        int na = n1v[a]; if (na < 0) na = 0; if (na > max_atoms1) na = max_atoms1;
        naq1_ll += 3ll * na;
    }
    for (int b = 0; b < nm2; ++b) {
        int nb = n2v[b]; if (nb < 0) nb = 0; if (nb > max_atoms2) nb = max_atoms2;
        naq2_ll += 3ll * nb;
    }
    if (naq1_ll < 0 || naq2_ll < 0) throw std::overflow_error("naq1/naq2 negative?");
    if (naq1_ll > std::numeric_limits<py::ssize_t>::max() ||
        naq2_ll > std::numeric_limits<py::ssize_t>::max())
        throw std::overflow_error("naq1/naq2 too large.");
    const int naq1 = static_cast<int>(naq1_ll);
    const int naq2 = static_cast<int>(naq2_ll);

    // If any side has zero derivatives, return an empty shaped array
    if (naq1 == 0 || naq2 == 0) {
        py::array_t<double> K(
            py::array::ShapeContainer{ (py::ssize_t)naq2, (py::ssize_t)naq1 },
            py::array::StridesContainer{ (py::ssize_t)(naq1 * sizeof(double)), (py::ssize_t)sizeof(double) }
        );
        return K; // NumPy owns a small empty buffer
    }

    // ---- aligned output allocation (naq2 x naq1), row-major ----
    double* Kptr = aligned_alloc_64((std::size_t)naq2 * naq1);
    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    py::array_t<double> K(
        /*shape*/   py::array::ShapeContainer{ (py::ssize_t)naq2, (py::ssize_t)naq1 },
        /*strides*/ py::array::StridesContainer{
            (py::ssize_t)(naq1 * sizeof(double)),
            (py::ssize_t)sizeof(double)
        },
        /*ptr*/     Kptr,
        /*base*/    capsule
    );

    // ---- call C++ (release GIL) ----
    {
        py::gil_scoped_release release;
        fchl19::fgdml_kernel(
            x1v, x2v, dx1v, dx2v, q1v, q2v, n1v, n2v,
            nm1, nm2, max_atoms1, max_atoms2, rep,
            naq1, naq2, sigma,
            Kptr
        );
    }

    return K;
}




PYBIND11_MODULE(_fchl19, m) {
    m.doc() = "Pybind11 bindings for FCHL-like ACSF generator";
        m.doc() = "FCHL19 local kernel (C++/pybind11)";
        m.def("fgdml_kernel", &fgdml_kernel_py,
          py::arg("x1"), py::arg("x2"),
          py::arg("dx1"), py::arg("dx2"),
          py::arg("q1"), py::arg("q2"),
          py::arg("n1"), py::arg("n2"),
          py::arg("sigma"),
          R"(Compute the FGDML Hessian kernel.

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

    m.def("fatomic_local_gradient_kernel", &fatomic_local_gradient_kernel_py,
          py::arg("x1"), py::arg("x2"), py::arg("dX2"),
          py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
          py::arg("sigma"),
          R"(Compute the gradient of the local kernel w.r.t. coordinates of set-2.

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


    m.def("compute_rep_size", &fchl19::compute_rep_size,
    py::arg("nelements"), py::arg("nbasis2"), py::arg("nbasis3"), py::arg("nabasis"),
    "Compute the total representation size per-atom.");
    
    
    m.def("generate_fchl_acsf", &generate_fchl_acsf_py,
        py::arg("coords"),
        py::arg("nuclear_z"),
        py::arg("elements") = std::vector<int>{1,6,7,8,16},
        py::arg("nRs2") = 24,
        py::arg("nRs3") = 20,
        py::arg("nFourier") = 1,
        py::arg("eta2") = 0.32,
        py::arg("eta3") = 2.7,
        py::arg("zeta") = M_PI,
        py::arg("rcut") = 8.0,
        py::arg("acut") = 8.0,
        py::arg("two_body_decay") = 1.8,
        py::arg("three_body_decay") = 0.57,
        py::arg("three_body_weight") = 13.4,
R"pbdoc(
Generate FCHL-like ACSF representation.


Parameters
----------
coords : ndarray, shape (n_atoms, 3) or (3*n_atoms,)
Cartesian coordinates.
nuclear_z : ndarray of int, shape (n_atoms,)
Nuclear charges.
elements : list[int], default [1,6,7,8,16]
Unique element types present.
nRs2 : int, default 24
nRs3 : int, default 20
nFourier : int, default 1
Rs2 = linspace(0, rcut, 1+nRs2)[1:]
Rs3 = linspace(0, acut, 1+nRs3)[1:]
Ts = linspace(0, pi, 2*nFourier)
eta2 : float, default 0.32
eta3 : float, default 2.7
zeta : float, default pi
rcut : float, default 8.0
acut : float, default 8.0
two_body_decay : float, default 1.8
three_body_decay : float, default 0.57
three_body_weight : float, default 13.4, rescaled internally by sqrt(eta3/pi)


Returns
-------
ndarray, shape (n_atoms, rep_size)
Per-atom representation.
)pbdoc");

        m.def("generate_fchl_acsf_and_gradients", &generate_fchl_acsf_rep_and_grad_py,
          py::arg("coords"),
          py::arg("nuclear_z"),
          py::arg("elements") = std::vector<int>{1,6,7,8,16},
          py::arg("nRs2") = 24,
          py::arg("nRs3") = 20,
          py::arg("nFourier") = 1,
          py::arg("eta2") = 0.32,
          py::arg("eta3") = 2.7,
          py::arg("zeta") = M_PI,
          py::arg("rcut") = 8.0,
          py::arg("acut") = 8.0,
          py::arg("two_body_decay") = 1.8,
          py::arg("three_body_decay") = 0.57,
          py::arg("three_body_weight") = 13.4,
          R"pbdoc(
Generate ACSF representation and its Jacobian with respect to atomic coordinates.

This version computes gradients by **central finite differences** over coordinates
(step size `h`, default 1e-6). It matches the representation built from internal
Rs2/Rs3/Ts (linspace) and rescales the three-body weight by sqrt(eta3/pi).

Returns
-------
(rep, grad)
  rep : (n_atoms, rep_size)
  grad: (n_atoms, rep_size, n_atoms, 3)
)pbdoc");
    
    m.def("flocal_kernel", &flocal_kernel_py,
        py::arg("x1"), py::arg("x2"),
        py::arg("q1"), py::arg("q2"),
        py::arg("n1"), py::arg("n2"),
        py::arg("sigma")
    );
    m.def("flocal_kernel_symm", &flocal_kernel_symm_py,
        py::arg("x1"),
        py::arg("q1"),
        py::arg("n1"),
        py::arg("sigma")
    );
}
