#pragma once
#include <cstddef>
#include <vector>

namespace kf {
namespace fchl18 {

// Generate FCHL18 representation for a single molecule.
//
// coords:       (n_atoms * 3) row-major Cartesian coordinates [Angstrom]
// nuclear_z:    (n_atoms) nuclear charges
// max_size:     padded number of atoms (output first dimension)
// cut_distance: neighbour cutoff radius [Angstrom]
//
// Output x: (max_size * 5 * max_size) row-major array, layout [atom, channel, neighbour]
//   channel 0 : sorted distances to neighbours (1e100 for padding)
//   channel 1 : nuclear charges of neighbours
//   channel 2 : dx (neighbour - centre)
//   channel 3 : dy
//   channel 4 : dz
//
// Output n_neighbors: (max_size) number of neighbours within cutoff for each atom.
//   Includes the atom itself at index 0 (distance 0, same Z, zero displacement).
//
// Fortran-equivalent: generate_fchl18 from fchl_representations.py
void generate_fchl18(
    const std::vector<double> &coords,  // (n_atoms, 3) row-major
    const std::vector<int> &nuclear_z,  // (n_atoms)
    int max_size, double cut_distance,
    std::vector<double> &x,        // (max_size, 5, max_size) row-major OUT
    std::vector<int> &n_neighbors  // (max_size) OUT
);

}  // namespace fchl18
}  // namespace kf
