#pragma once

// FCHL18 periodic-table alchemical overlap (qmllib / qml compatibility).
//
// pd[Zi-1, Zj-1] = exp( -(P_i-P_j)^2 / (4 r_w^2) - (G_i-G_j)^2 / (4 c_w^2) )
// with (P,G) = period/group from the same PTP table as qmllib.utils.alchemy.
//
// qml's get_local_kernels passes:
//   r_width = alchemy_group_width,  c_width = alchemy_period_width
// (names are swapped relative to "period"/"group" — match qml exactly).

#include <cmath>
#include <stdexcept>
#include <vector>

namespace kf {
namespace fchl18 {

inline constexpr int kAlchemyEmaxDefault = 100;

// Period / group for Z = 1 .. 118. Unlisted Z → {0,0} (pd → 0 vs anything with P,G>0).
inline void period_group(int z, int &period, int &group) {
    period = 0;
    group = 0;
    // clang-format off
    switch (z) {
        case 1:  period = 1; group = 1;  break;
        case 2:  period = 1; group = 8;  break;
        case 3:  period = 2; group = 1;  break;
        case 4:  period = 2; group = 2;  break;
        case 5:  period = 2; group = 3;  break;
        case 6:  period = 2; group = 4;  break;
        case 7:  period = 2; group = 5;  break;
        case 8:  period = 2; group = 6;  break;
        case 9:  period = 2; group = 7;  break;
        case 10: period = 2; group = 8;  break;
        case 11: period = 3; group = 1;  break;
        case 12: period = 3; group = 2;  break;
        case 13: period = 3; group = 3;  break;
        case 14: period = 3; group = 4;  break;
        case 15: period = 3; group = 5;  break;
        case 16: period = 3; group = 6;  break;
        case 17: period = 3; group = 7;  break;
        case 18: period = 3; group = 8;  break;
        case 19: period = 4; group = 1;  break;
        case 20: period = 4; group = 2;  break;
        case 21: period = 4; group = 9;  break;
        case 22: period = 4; group = 10; break;
        case 23: period = 4; group = 11; break;
        case 24: period = 4; group = 12; break;
        case 25: period = 4; group = 13; break;
        case 26: period = 4; group = 14; break;
        case 27: period = 4; group = 15; break;
        case 28: period = 4; group = 16; break;
        case 29: period = 4; group = 17; break;
        case 30: period = 4; group = 18; break;
        case 31: period = 4; group = 3;  break;
        case 32: period = 4; group = 4;  break;
        case 33: period = 4; group = 5;  break;
        case 34: period = 4; group = 6;  break;
        case 35: period = 4; group = 7;  break;
        case 36: period = 4; group = 8;  break;
        case 37: period = 5; group = 1;  break;
        case 38: period = 5; group = 2;  break;
        case 39: period = 5; group = 9;  break;
        case 40: period = 5; group = 10; break;
        case 41: period = 5; group = 11; break;
        case 42: period = 5; group = 12; break;
        case 43: period = 5; group = 13; break;
        case 44: period = 5; group = 14; break;
        case 45: period = 5; group = 15; break;
        case 46: period = 5; group = 16; break;
        case 47: period = 5; group = 17; break;
        case 48: period = 5; group = 18; break;
        case 49: period = 5; group = 3;  break;
        case 50: period = 5; group = 4;  break;
        case 51: period = 5; group = 5;  break;
        case 52: period = 5; group = 6;  break;
        case 53: period = 5; group = 7;  break;
        case 54: period = 5; group = 8;  break;
        case 55: period = 6; group = 1;  break;
        case 56: period = 6; group = 2;  break;
        case 57: period = 6; group = 19; break;
        case 58: period = 6; group = 20; break;
        case 59: period = 6; group = 21; break;
        case 60: period = 6; group = 22; break;
        case 61: period = 6; group = 23; break;
        case 62: period = 6; group = 24; break;
        case 63: period = 6; group = 25; break;
        case 64: period = 6; group = 26; break;
        case 65: period = 6; group = 27; break;
        case 66: period = 6; group = 28; break;
        case 67: period = 6; group = 29; break;
        case 68: period = 6; group = 30; break;
        case 69: period = 6; group = 31; break;
        case 70: period = 6; group = 32; break;
        case 71: period = 6; group = 33; break;
        case 72: period = 6; group = 10; break;
        case 73: period = 6; group = 11; break;
        case 74: period = 6; group = 12; break;
        case 75: period = 6; group = 13; break;
        case 76: period = 6; group = 14; break;
        case 77: period = 6; group = 15; break;
        case 78: period = 6; group = 16; break;
        case 79: period = 6; group = 17; break;
        case 80: period = 6; group = 18; break;
        case 81: period = 6; group = 3;  break;
        case 82: period = 6; group = 4;  break;
        case 83: period = 6; group = 5;  break;
        case 84: period = 6; group = 6;  break;
        case 85: period = 6; group = 7;  break;
        case 86: period = 6; group = 8;  break;
        case 87: period = 7; group = 1;  break;
        case 88: period = 7; group = 2;  break;
        case 89: period = 7; group = 19; break;
        case 90: period = 7; group = 20; break;
        case 91: period = 7; group = 21; break;
        case 92: period = 7; group = 22; break;
        case 93: period = 7; group = 23; break;
        case 94: period = 7; group = 24; break;
        case 95: period = 7; group = 25; break;
        case 96: period = 7; group = 26; break;
        case 97: period = 7; group = 27; break;
        case 98: period = 7; group = 28; break;
        case 99: period = 7; group = 29; break;
        case 100: period = 7; group = 30; break;
        case 101: period = 7; group = 32; break;
        case 102: period = 7; group = 14; break;
        case 103: period = 7; group = 33; break;
        case 104: period = 7; group = 10; break;
        case 105: period = 7; group = 11; break;
        case 106: period = 7; group = 12; break;
        case 107: period = 7; group = 13; break;
        case 108: period = 7; group = 14; break;
        case 109: period = 7; group = 15; break;
        case 110: period = 7; group = 16; break;
        case 111: period = 7; group = 17; break;
        case 112: period = 7; group = 18; break;
        case 113: period = 7; group = 3;  break;
        case 114: period = 7; group = 4;  break;
        case 115: period = 7; group = 5;  break;
        case 116: period = 7; group = 6;  break;
        case 117: period = 7; group = 7;  break;
        case 118: period = 7; group = 8;  break;
        default: break;
    }
    // clang-format on
}

// qml-compatible widths: r_width = alchemy_group_width, c_width = alchemy_period_width.
inline double periodic_table_overlap(
    int za, int zb, double alchemy_period_width, double alchemy_group_width
) {
    const double r_width = alchemy_group_width;
    const double c_width = alchemy_period_width;
    int pa = 0, ga = 0, pb = 0, gb = 0;
    period_group(za, pa, ga);
    period_group(zb, pb, gb);
    const double dp = static_cast<double>(pa - pb);
    const double dg = static_cast<double>(ga - gb);
    return std::exp(
        -(dp * dp) / (4.0 * r_width * r_width) - (dg * dg) / (4.0 * c_width * c_width)
    );
}

// Dense pd matrix, row-major, size emax*emax, 0-based Z-1 indexing (Z=1 → row 0).
inline std::vector<double> build_periodic_table_pd(
    int emax, double alchemy_period_width, double alchemy_group_width
) {
    if (emax <= 0) throw std::invalid_argument("alchemy emax must be > 0");
    if (alchemy_period_width <= 0.0 || alchemy_group_width <= 0.0) {
        throw std::invalid_argument("alchemy widths must be > 0");
    }
    std::vector<double> pd(static_cast<std::size_t>(emax) * emax, 0.0);
    for (int i = 0; i < emax; ++i) {
        for (int j = 0; j < emax; ++j) {
            pd[static_cast<std::size_t>(i) * emax + j] = periodic_table_overlap(
                i + 1, j + 1, alchemy_period_width, alchemy_group_width
            );
        }
    }
    return pd;
}

inline double pd_lookup(const double *pd, int emax, int za, int zb) {
    if (pd == nullptr || za < 1 || zb < 1 || za > emax || zb > emax) return 0.0;
    return pd[static_cast<std::size_t>(za - 1) * emax + (zb - 1)];
}

}  // namespace fchl18
}  // namespace kf
