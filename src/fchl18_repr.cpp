// Own header
#include "fchl18_repr.hpp"

// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kf {
namespace fchl18 {

void generate_fchl18(
    const std::vector<double> &coords,    // (n_atoms * 3) row-major
    const std::vector<int>    &nuclear_z, // (n_atoms)
    int                        max_size,
    double                     cut_distance,
    std::vector<double>       &x,         // (max_size * 5 * max_size) row-major OUT
    std::vector<int>          &n_neighbors // (max_size) OUT
) {
    const int n_atoms = static_cast<int>(nuclear_z.size());

    if (n_atoms == 0) throw std::invalid_argument("n_atoms must be > 0");
    if (max_size < n_atoms)
        throw std::invalid_argument("max_size must be >= n_atoms");
    if (cut_distance <= 0.0)
        throw std::invalid_argument("cut_distance must be > 0");
    if (static_cast<int>(coords.size()) != n_atoms * 3)
        throw std::invalid_argument("coords size mismatch");

    // Allocate output: (max_size, 5, max_size), all initialised to 1e100
    const std::size_t total = static_cast<std::size_t>(max_size) * 5 * max_size;
    x.assign(total, 1e100);
    n_neighbors.assign(max_size, 0);

    // Helper: x[(atom, channel, neighbour)] = x[atom*5*max_size + channel*max_size + neighbour]
    const int stride_atom = 5 * max_size;
    const int stride_chan = max_size;

    auto xidx = [&](int atom, int chan, int neigh) -> std::size_t {
        return static_cast<std::size_t>(atom) * stride_atom
             + static_cast<std::size_t>(chan)  * stride_chan
             + neigh;
    };

    // For each centre atom i, find all atoms j within cut_distance, sort by distance
    for (int i = 0; i < n_atoms; ++i) {
        const double xi = coords[3 * i + 0];
        const double yi = coords[3 * i + 1];
        const double zi = coords[3 * i + 2];

        // Collect (distance, index) pairs for all atoms (including self at dist 0)
        std::vector<std::pair<double, int>> nbrs;
        nbrs.reserve(n_atoms);

        for (int j = 0; j < n_atoms; ++j) {
            const double dx = coords[3 * j + 0] - xi;
            const double dy = coords[3 * j + 1] - yi;
            const double dz = coords[3 * j + 2] - zi;
            const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < cut_distance) {
                nbrs.emplace_back(r, j);
            }
        }

        // Sort by distance
        std::sort(nbrs.begin(), nbrs.end(),
                  [](const std::pair<double,int> &a, const std::pair<double,int> &b){
                      return a.first < b.first;
                  });

        const int nn = static_cast<int>(nbrs.size());
        // Clamp to max_size neighbours
        const int nn_use = std::min(nn, max_size);
        n_neighbors[i] = nn_use;

        for (int k = 0; k < nn_use; ++k) {
            const double r  = nbrs[k].first;
            const int    j  = nbrs[k].second;
            const double dx = coords[3 * j + 0] - xi;
            const double dy = coords[3 * j + 1] - yi;
            const double dz = coords[3 * j + 2] - zi;

            x[xidx(i, 0, k)] = r;
            x[xidx(i, 1, k)] = static_cast<double>(nuclear_z[j]);
            x[xidx(i, 2, k)] = dx;
            x[xidx(i, 3, k)] = dy;
            x[xidx(i, 4, k)] = dz;
        }
    }
}

}  // namespace fchl18
}  // namespace kf
