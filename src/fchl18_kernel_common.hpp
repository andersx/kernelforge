#pragma once

// Shared helpers for FCHL18 kernel binding functions.
//
// Defines:
//   - MolData struct  (n_atoms, coords (n*3), z (n))
//   - parse_mol()     converts a (n,3) coords array + (n,) z array into MolData
//
// Include this header in any .cpp that needs to parse molecule lists from Python.

#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kf {
namespace fchl18 {

// ---------------------------------------------------------------------------
// MolData: parsed representation of a single molecule
// ---------------------------------------------------------------------------
struct MolData {
    int                 n_atoms;  // number of atoms
    std::vector<double> coords;   // flat (n_atoms * 3), row-major
    std::vector<int>    z;        // nuclear charges (n_atoms)
};

// ---------------------------------------------------------------------------
// parse_mol: convert pybind11 objects to MolData
//
// c_obj : (n_atoms, 3) float64 array — Cartesian coordinates
// z_obj : (n_atoms,)   int32  array — nuclear charges
// ---------------------------------------------------------------------------
inline MolData parse_mol(const py::object &c_obj, const py::object &z_obj) {
    auto c = py::array_t<double,  py::array::c_style | py::array::forcecast>::ensure(c_obj);
    auto z = py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(z_obj);
    if (!c || c.ndim() != 2 || c.shape(1) != 3)
        throw std::invalid_argument("each coords array must have shape (n_atoms, 3)");
    if (!z || z.ndim() != 1 || z.shape(0) != c.shape(0))
        throw std::invalid_argument("each z array must have shape (n_atoms,)");
    const int na = static_cast<int>(c.shape(0));
    MolData md;
    md.n_atoms = na;
    md.coords.resize(static_cast<std::size_t>(na) * 3);
    md.z.resize(na);
    auto cr = c.unchecked<2>();
    auto zr = z.unchecked<1>();
    for (int i = 0; i < na; ++i) {
        md.coords[i*3+0] = cr(i, 0);
        md.coords[i*3+1] = cr(i, 1);
        md.coords[i*3+2] = cr(i, 2);
        md.z[i] = static_cast<int>(zr(i));
    }
    return md;
}

}  // namespace fchl18
}  // namespace kf
