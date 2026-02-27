// C++ standard library
#include <stdexcept>
#include <vector>

// Third-party libraries
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Project headers
#include "fchl18_repr.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Generate FCHL18 representation for a batch of molecules.
//
// coords_list   : list of (n_i, 3) float64 arrays  — one per molecule
// nuclear_z_list: list of (n_i,)   int32  arrays  — one per molecule
// max_size      : maximum number of atoms (padding dimension)
// cut_distance  : neighbour cutoff
//
// Returns a tuple (x, n_atoms, n_neighbors):
//   x           : (nm, max_size, 5, max_size)  float64, row-major
//   n_atoms     : (nm,)                         int32
//   n_neighbors : (nm, max_size)                int32
// ---------------------------------------------------------------------------
static py::tuple generate_fchl18_batch_py(
    const py::list &coords_list,
    const py::list &nuclear_z_list,
    int    max_size,
    double cut_distance
) {
    const int nm = static_cast<int>(coords_list.size());
    if (nm == 0) throw std::invalid_argument("coords_list must not be empty");
    if (static_cast<int>(nuclear_z_list.size()) != nm)
        throw std::invalid_argument("coords_list and nuclear_z_list must have the same length");
    if (max_size <= 0) throw std::invalid_argument("max_size must be > 0");
    if (cut_distance <= 0.0) throw std::invalid_argument("cut_distance must be > 0");

    // Allocate output arrays
    py::array_t<double> x_out(
        {(py::ssize_t)nm, (py::ssize_t)max_size, (py::ssize_t)5, (py::ssize_t)max_size}
    );
    py::array_t<int32_t> n_atoms_out({(py::ssize_t)nm});
    py::array_t<int32_t> n_neighbors_out({(py::ssize_t)nm, (py::ssize_t)max_size});

    auto x_buf   = x_out.mutable_unchecked<4>();
    auto na_buf  = n_atoms_out.mutable_unchecked<1>();
    auto nn_buf  = n_neighbors_out.mutable_unchecked<2>();

    // Initialise x to 1e100 (padding value)
    {
        double *ptr = x_out.mutable_data();
        const std::size_t total = (std::size_t)nm * max_size * 5 * max_size;
        std::fill(ptr, ptr + total, 1e100);
    }
    // Initialise counts to 0
    std::fill(n_atoms_out.mutable_data(), n_atoms_out.mutable_data() + nm, 0);
    std::fill(n_neighbors_out.mutable_data(), n_neighbors_out.mutable_data() + nm * max_size, 0);

    for (int a = 0; a < nm; ++a) {
        // Parse coords: (n_a, 3) or (3*n_a,)
        auto coords_arr = py::cast<py::array>(coords_list[a])
                              .cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        std::size_t n_atoms_a = 0;
        std::vector<double> coords_v;

        if (coords_arr.ndim() == 2) {
            n_atoms_a = static_cast<std::size_t>(coords_arr.shape(0));
            if (coords_arr.shape(1) != 3)
                throw std::invalid_argument("coords must have shape (n_atoms, 3)");
            coords_v.resize(n_atoms_a * 3);
            auto r = coords_arr.unchecked<2>();
            for (std::size_t i = 0; i < n_atoms_a; ++i) {
                coords_v[3*i+0] = r(i, 0);
                coords_v[3*i+1] = r(i, 1);
                coords_v[3*i+2] = r(i, 2);
            }
        } else if (coords_arr.ndim() == 1) {
            if (coords_arr.shape(0) % 3 != 0)
                throw std::invalid_argument("1D coords length must be divisible by 3");
            n_atoms_a = static_cast<std::size_t>(coords_arr.shape(0)) / 3;
            auto r = coords_arr.unchecked<1>();
            coords_v.assign(r.data(0), r.data(0) + n_atoms_a * 3);
        } else {
            throw std::invalid_argument("coords must be a 1D or 2D array");
        }

        if (static_cast<int>(n_atoms_a) > max_size)
            throw std::invalid_argument("n_atoms > max_size for molecule " + std::to_string(a));

        // Parse nuclear_z
        auto z_arr = py::cast<py::array>(nuclear_z_list[a])
                         .cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        if (z_arr.ndim() != 1 || static_cast<std::size_t>(z_arr.shape(0)) != n_atoms_a)
            throw std::invalid_argument("nuclear_z length must equal n_atoms for molecule "
                                        + std::to_string(a));
        auto zr = z_arr.unchecked<1>();
        std::vector<int> z_v(zr.data(0), zr.data(0) + n_atoms_a);

        na_buf(a) = static_cast<int32_t>(n_atoms_a);

        // Call C++ core
        std::vector<double> x_mol;
        std::vector<int>    nn_mol;
        {
            py::gil_scoped_release release;
            kf::fchl18::generate_fchl18(
                coords_v, z_v, max_size, cut_distance, x_mol, nn_mol
            );
        }

        // Copy x_mol (max_size, 5, max_size) into x_out[a, :, :, :]
        for (int i = 0; i < max_size; ++i)
            for (int c = 0; c < 5; ++c)
                for (int k = 0; k < max_size; ++k)
                    x_buf(a, i, c, k) = x_mol[
                        static_cast<std::size_t>(i) * 5 * max_size
                      + static_cast<std::size_t>(c) * max_size
                      + k
                    ];

        // Copy n_neighbors
        for (int i = 0; i < max_size; ++i)
            nn_buf(a, i) = static_cast<int32_t>(nn_mol[i]);
    }

    return py::make_tuple(x_out, n_atoms_out, n_neighbors_out);
}

PYBIND11_MODULE(fchl18_repr, m) {
    m.doc() = "FCHL18 representation generation";

    m.def(
        "generate",
        &generate_fchl18_batch_py,
        py::arg("coords_list"),
        py::arg("nuclear_z_list"),
        py::arg("max_size") = 23,
        py::arg("cut_distance") = 5.0,
        R"pbdoc(
Generate FCHL18 representations for a batch of molecules.

Parameters
----------
coords_list : list of ndarray, each shape (n_i, 3)
    Cartesian coordinates per molecule [Angstrom].
nuclear_z_list : list of ndarray of int, each shape (n_i,)
    Nuclear charges per molecule.
max_size : int, default 23
    Maximum number of atoms (padding dimension).
cut_distance : float, default 5.0
    Neighbour cutoff radius [Angstrom].

Returns
-------
x : ndarray, shape (nm, max_size, 5, max_size), float64
    Representation array. Layout per atom: [distances, Z-neighbours, dx, dy, dz].
    Padded with 1e100 for unused slots.
n_atoms : ndarray, shape (nm,), int32
    Number of real atoms in each molecule.
n_neighbors : ndarray, shape (nm, max_size), int32
    Number of neighbours within cut_distance for each atom (including self at index 0).
)pbdoc"
    );
}
