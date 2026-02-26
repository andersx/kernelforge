// C++ standard library
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "aligned_alloc64.hpp"
#include "rff_elemental.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Shared helper: unpack Q list + build Q_flat/sizes, return total_atoms
// ---------------------------------------------------------------------------
static std::size_t unpack_Q(
    const py::list &Q_list, std::size_t nmol, std::size_t max_atoms, std::vector<int> &Q_flat,
    std::vector<int> &sizes
) {
    Q_flat.assign(nmol * max_atoms, -1);
    sizes.resize(nmol);
    std::size_t total_atoms = 0;

    for (std::size_t i = 0; i < nmol; ++i) {
        auto q_arr = Q_list[i].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        const auto natoms = static_cast<std::size_t>(q_arr.shape(0));
        if (natoms > max_atoms)
            throw std::invalid_argument(
                "Q[" + std::to_string(i) + "] has more atoms than X.shape[1]"
            );
        sizes[i] = static_cast<int>(natoms);
        total_atoms += natoms;
        const int *qdata = q_arr.data();
        for (std::size_t j = 0; j < natoms; ++j) {
            Q_flat[i * max_atoms + j] = qdata[j];
        }
    }
    return total_atoms;
}

// ---- rff_features_elemental -------------------------------------------------

static py::array_t<double> py_rff_features_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(W_arr.shape(1)) != rep_size)
        throw std::invalid_argument("W.shape[1] must equal X.shape[2] (rep_size)");
    if (static_cast<std::size_t>(W_arr.shape(2)) != D)
        throw std::invalid_argument("W.shape[2] must equal b.shape[1] (D)");
    if (static_cast<std::size_t>(b_arr.shape(0)) != nelements)
        throw std::invalid_argument("b.shape[0] must equal W.shape[0] (nelements)");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    double *LZptr = aligned_alloc_64(nmol * D);
    auto capsule = py::capsule(LZptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_features_elemental(
        X_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        LZptr
    );

    return py::array_t<double>(
        {static_cast<py::ssize_t>(nmol), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        LZptr,
        capsule
    );
}

// ---- rff_gradient_elemental -------------------------------------------------

static py::array_t<double> py_rff_gradient_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != nmol ||
        static_cast<std::size_t>(dX_arr.shape(1)) != max_atoms ||
        static_cast<std::size_t>(dX_arr.shape(2)) != rep_size ||
        static_cast<std::size_t>(dX_arr.shape(3)) != max_atoms ||
        static_cast<std::size_t>(dX_arr.shape(4)) != 3)
        throw std::invalid_argument("dX shape must be (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;

    double *Gptr = aligned_alloc_64(D * ngrads);
    auto capsule = py::capsule(Gptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient_elemental(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        ngrads,
        Gptr
    );

    return py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(ngrads)},
        {static_cast<py::ssize_t>(ngrads * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))
        },
        Gptr,
        capsule
    );
}

// ---- rff_full_elemental -----------------------------------------------------

static py::array_t<double> py_rff_full_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(dX_arr.shape(0)) != nmol ||
        static_cast<std::size_t>(dX_arr.shape(1)) != max_atoms ||
        static_cast<std::size_t>(dX_arr.shape(2)) != rep_size ||
        static_cast<std::size_t>(dX_arr.shape(3)) != max_atoms ||
        static_cast<std::size_t>(dX_arr.shape(4)) != 3)
        throw std::invalid_argument("dX shape must be (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;
    const std::size_t total_rows = nmol + ngrads;

    double *Zptr = aligned_alloc_64(total_rows * D);
    auto capsule = py::capsule(Zptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full_elemental(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        ngrads,
        Zptr
    );

    return py::array_t<double>(
        {static_cast<py::ssize_t>(total_rows), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        Zptr,
        capsule
    );
}

// ---- rff_gramian_elemental --------------------------------------------------

static py::tuple py_rff_gramian_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    std::size_t chunk_size
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (Y_arr.ndim() != 1) throw std::invalid_argument("Y must be 1D (nmol,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(Y_arr.shape(0)) != nmol)
        throw std::invalid_argument("Y.shape[0] must equal nmol");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    double *ZtZ_ptr = aligned_alloc_64(D * D);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(ZtZ_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gramian_elemental(
        X_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        Y_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        chunk_size,
        ZtZ_ptr,
        ZtY_ptr
    );

    auto ZtZ_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_ptr,
        cap_gram
    );

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr,
        cap_proj
    );

    return py::make_tuple(ZtZ_out, ZtY_out);
}

// ---- rff_gradient_gramian_elemental -----------------------------------------

static py::tuple py_rff_gradient_gramian_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t chunk_size
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (F_arr.ndim() != 1) throw std::invalid_argument("F must be 1D (ngrads,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;
    if (static_cast<std::size_t>(F_arr.shape(0)) != ngrads)
        throw std::invalid_argument(
            "F.shape[0] must equal 3 * sum(atom counts) = " + std::to_string(ngrads)
        );

    double *GtG_ptr = aligned_alloc_64(D * D);
    double *GtF_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(GtG_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(GtF_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient_gramian_elemental(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        F_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        chunk_size,
        GtG_ptr,
        GtF_ptr
    );

    auto GtG_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        GtG_ptr,
        cap_gram
    );

    auto GtF_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtF_ptr,
        cap_proj
    );

    return py::make_tuple(GtG_out, GtF_out);
}

// ---- rff_full_gramian_elemental ---------------------------------------------

static py::tuple py_rff_full_gramian_elemental(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t energy_chunk, std::size_t force_chunk
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (Y_arr.ndim() != 1) throw std::invalid_argument("Y must be 1D (nmol,)");
    if (F_arr.ndim() != 1) throw std::invalid_argument("F must be 1D (ngrads,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(Y_arr.shape(0)) != nmol)
        throw std::invalid_argument("Y.shape[0] must equal nmol");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;
    if (static_cast<std::size_t>(F_arr.shape(0)) != ngrads)
        throw std::invalid_argument(
            "F.shape[0] must equal 3 * sum(atom counts) = " + std::to_string(ngrads)
        );

    double *ZtZ_ptr = aligned_alloc_64(D * D);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_gram = py::capsule(ZtZ_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full_gramian_elemental(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        Y_arr.data(),
        F_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        energy_chunk,
        force_chunk,
        ZtZ_ptr,
        ZtY_ptr
    );

    auto ZtZ_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(D * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_ptr,
        cap_gram
    );

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr,
        cap_proj
    );

    return py::make_tuple(ZtZ_out, ZtY_out);
}

// ---- rff_gramian_elemental_rfp ----------------------------------------------

static py::tuple py_rff_gramian_elemental_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    std::size_t chunk_size
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (Y_arr.ndim() != 1) throw std::invalid_argument("Y must be 1D (nmol,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(Y_arr.shape(0)) != nmol)
        throw std::invalid_argument("Y.shape[0] must equal nmol");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t nt = D * (D + 1) / 2;
    double *ZtZ_rfp_ptr = aligned_alloc_64(nt);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_rfp = py::capsule(ZtZ_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gramian_elemental_rfp(
        X_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        Y_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        chunk_size,
        ZtZ_rfp_ptr,
        ZtY_ptr
    );

    auto ZtZ_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_rfp_ptr,
        cap_rfp
    );

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr,
        cap_proj
    );

    return py::make_tuple(ZtZ_rfp_out, ZtY_out);
}

// ---- rff_gradient_gramian_elemental_rfp -------------------------------------

static py::tuple py_rff_gradient_gramian_elemental_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t chunk_size
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (F_arr.ndim() != 1) throw std::invalid_argument("F must be 1D (ngrads,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;
    if (static_cast<std::size_t>(F_arr.shape(0)) != ngrads)
        throw std::invalid_argument(
            "F.shape[0] must equal 3 * sum(atom counts) = " + std::to_string(ngrads)
        );

    const std::size_t nt = D * (D + 1) / 2;
    double *GtG_rfp_ptr = aligned_alloc_64(nt);
    double *GtF_ptr = aligned_alloc_64(D);
    auto cap_rfp = py::capsule(GtG_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(GtF_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_gradient_gramian_elemental_rfp(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        F_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        chunk_size,
        GtG_rfp_ptr,
        GtF_ptr
    );

    auto GtG_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtG_rfp_ptr,
        cap_rfp
    );

    auto GtF_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        GtF_ptr,
        cap_proj
    );

    return py::make_tuple(GtG_rfp_out, GtF_out);
}

// ---- rff_full_gramian_elemental_rfp -----------------------------------------

static py::tuple py_rff_full_gramian_elemental_rfp(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &X_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &dX_arr,
    const py::list &Q_list,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &W_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &b_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &Y_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &F_arr,
    std::size_t energy_chunk, std::size_t force_chunk
) {
    if (X_arr.ndim() != 3) throw std::invalid_argument("X must be 3D (nmol, max_atoms, rep_size)");
    if (dX_arr.ndim() != 5)
        throw std::invalid_argument("dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    if (W_arr.ndim() != 3) throw std::invalid_argument("W must be 3D (nelements, rep_size, D)");
    if (b_arr.ndim() != 2) throw std::invalid_argument("b must be 2D (nelements, D)");
    if (Y_arr.ndim() != 1) throw std::invalid_argument("Y must be 1D (nmol,)");
    if (F_arr.ndim() != 1) throw std::invalid_argument("F must be 1D (ngrads,)");

    const auto nmol = static_cast<std::size_t>(X_arr.shape(0));
    const auto max_atoms = static_cast<std::size_t>(X_arr.shape(1));
    const auto rep_size = static_cast<std::size_t>(X_arr.shape(2));
    const auto nelements = static_cast<std::size_t>(W_arr.shape(0));
    const auto D = static_cast<std::size_t>(b_arr.shape(1));

    if (static_cast<std::size_t>(Y_arr.shape(0)) != nmol)
        throw std::invalid_argument("Y.shape[0] must equal nmol");
    if (Q_list.size() != nmol) throw std::invalid_argument("len(Q) must equal X.shape[0]");

    std::vector<int> Q_flat, sizes;
    const std::size_t total_atoms = unpack_Q(Q_list, nmol, max_atoms, Q_flat, sizes);

    const std::size_t ngrads = 3 * total_atoms;
    if (static_cast<std::size_t>(F_arr.shape(0)) != ngrads)
        throw std::invalid_argument(
            "F.shape[0] must equal 3 * sum(atom counts) = " + std::to_string(ngrads)
        );

    const std::size_t nt = D * (D + 1) / 2;
    double *ZtZ_rfp_ptr = aligned_alloc_64(nt);
    double *ZtY_ptr = aligned_alloc_64(D);
    auto cap_rfp = py::capsule(ZtZ_rfp_ptr, [](void *p) { aligned_free_64(p); });
    auto cap_proj = py::capsule(ZtY_ptr, [](void *p) { aligned_free_64(p); });

    kf::rff::rff_full_gramian_elemental_rfp(
        X_arr.data(),
        dX_arr.data(),
        Q_flat.data(),
        sizes.data(),
        W_arr.data(),
        b_arr.data(),
        Y_arr.data(),
        F_arr.data(),
        nmol,
        max_atoms,
        rep_size,
        nelements,
        D,
        energy_chunk,
        force_chunk,
        ZtZ_rfp_ptr,
        ZtY_ptr
    );

    auto ZtZ_rfp_out = py::array_t<double>(
        {static_cast<py::ssize_t>(nt)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtZ_rfp_ptr,
        cap_rfp
    );

    auto ZtY_out = py::array_t<double>(
        {static_cast<py::ssize_t>(D)},
        {static_cast<py::ssize_t>(sizeof(double))},
        ZtY_ptr,
        cap_proj
    );

    return py::make_tuple(ZtZ_rfp_out, ZtY_out);
}

// ---- Registration entry point -----------------------------------------------

void register_rff_elemental(py::module_ &m) {
    m.def(
        "rff_features_elemental",
        &py_rff_features_elemental,
        R"doc(
Compute element-stratified Random Fourier Features with per-molecule summation.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)

Returns
-------
LZ : ndarray, shape (nmol, D)
)doc",
        py::arg("X"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b")
    );

    m.def(
        "rff_gradient_elemental",
        &py_rff_gradient_elemental,
        R"doc(
Compute gradient of element-stratified RFF features w.r.t. atomic coordinates.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)

Returns
-------
G : ndarray, shape (D, ngrads)  where ngrads = 3 * sum(sizes)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b")
    );

    m.def(
        "rff_full_elemental",
        &py_rff_full_elemental,
        R"doc(
Compute combined energy+force elemental RFF feature matrix (stacked).

Z_full[0:nmol, :]           = rff_features_elemental(...)    (nmol, D)
Z_full[nmol:nmol+ngrads, :] = rff_gradient_elemental(...).T  (ngrads, D)

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)

Returns
-------
Z_full : ndarray, shape (nmol + ngrads, D)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b")
    );

    m.def(
        "rff_gramian_elemental",
        &py_rff_gramian_elemental,
        R"doc(
Compute chunked Gramian for energy-only elemental RFF training.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
Y : ndarray, shape (nmol,)
chunk_size : int

Returns
-------
ZtZ : ndarray, shape (D, D)
ZtY : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("chunk_size") = 8192
    );

    m.def(
        "rff_gradient_gramian_elemental",
        &py_rff_gradient_gramian_elemental,
        R"doc(
Compute chunked Gramian for force-only elemental RFF training.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
F : ndarray, shape (ngrads,)
chunk_size : int

Returns
-------
GtG : ndarray, shape (D, D)
GtF : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("F"),
        py::arg("chunk_size") = 256
    );

    m.def(
        "rff_full_gramian_elemental",
        &py_rff_full_gramian_elemental,
        R"doc(
Compute chunked Gramian for energy+force elemental RFF training.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
Y : ndarray, shape (nmol,)
F : ndarray, shape (ngrads,)
energy_chunk : int
force_chunk : int

Returns
-------
ZtZ : ndarray, shape (D, D)
ZtY : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("F"),
        py::arg("energy_chunk") = 8192,
        py::arg("force_chunk") = 256
    );

    m.def(
        "rff_gramian_elemental_rfp",
        &py_rff_gramian_elemental_rfp,
        R"doc(
Compute chunked Gramian for energy-only elemental RFF training, RFP-packed output.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
Y : ndarray, shape (nmol,)
chunk_size : int

Returns
-------
ZtZ_rfp : ndarray, shape (D*(D+1)//2,)
ZtY : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("chunk_size") = 8192
    );

    m.def(
        "rff_gradient_gramian_elemental_rfp",
        &py_rff_gradient_gramian_elemental_rfp,
        R"doc(
Compute chunked Gramian for force-only elemental RFF training, RFP-packed output.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
F : ndarray, shape (ngrads,)
chunk_size : int

Returns
-------
GtG_rfp : ndarray, shape (D*(D+1)//2,)
GtF : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("F"),
        py::arg("chunk_size") = 256
    );

    m.def(
        "rff_full_gramian_elemental_rfp",
        &py_rff_full_gramian_elemental_rfp,
        R"doc(
Compute chunked Gramian for energy+force elemental RFF training, RFP-packed output.

Parameters
----------
X : ndarray, shape (nmol, max_atoms, rep_size)
dX : ndarray, shape (nmol, max_atoms, rep_size, max_atoms, 3)
Q : list of ndarray[int]
W : ndarray, shape (nelements, rep_size, D)
b : ndarray, shape (nelements, D)
Y : ndarray, shape (nmol,)
F : ndarray, shape (ngrads,)
energy_chunk : int
force_chunk : int

Returns
-------
ZtZ_rfp : ndarray, shape (D*(D+1)//2,)
ZtY : ndarray, shape (D,)
)doc",
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("F"),
        py::arg("energy_chunk") = 8192,
        py::arg("force_chunk") = 256
    );
}
