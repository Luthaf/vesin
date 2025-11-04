#ifndef VESIN_TYPES_HPP
#define VESIN_TYPES_HPP

#include <array>
#include <cassert>

#include "math.hpp"

namespace vesin {

class BoundingBox {
public:
    BoundingBox(Matrix matrix, bool periodic[3]):
        matrix_(matrix),
        periodic_({periodic[0], periodic[1], periodic[2]}) {

        // find number of periodic directions and their indices
        int n_periodic = 0;
        int periodic_idx_1 = -1;
        int periodic_idx_2 = -1;
        for (int i = 0; i < 3; ++i) {
            if (periodic_[i]) {
                n_periodic += 1;
                if (periodic_idx_1 == -1) {
                    periodic_idx_1 = i;
                } else if (periodic_idx_2 == -1) {
                    periodic_idx_2 = i;
                }
            }
        }

        // adjust the box matrix to have a simple orthogonal dimension along
        // non-periodic directions
        if (n_periodic == 0) {
            matrix_ = Matrix{
                std::array<double, 3>{1, 0, 0},
                std::array<double, 3>{0, 1, 0},
                std::array<double, 3>{0, 0, 1},
            };
        } else if (n_periodic == 1) {
            assert(periodic_idx_1 != -1);
            // Make the two non-periodic directions orthogonal to the periodic one
            auto a = Vector{matrix_[periodic_idx_1]};
            auto b = Vector{0, 1, 0};
            if (std::abs(a.normalize().dot(b)) > 0.9) {
                b = Vector{0, 0, 1};
            }
            auto c = a.cross(b).normalize();
            b = c.cross(a).normalize();

            // Assign back to the matrix picking the "non-periodic" indices without ifs
            matrix_[(periodic_idx_1 + 1) % 3] = b;
            matrix_[(periodic_idx_1 + 2) % 3] = c;
        } else if (n_periodic == 2) {
            assert(periodic_idx_1 != -1 && periodic_idx_2 != -1);
            // Make the one non-periodic direction orthogonal to the two periodic ones
            auto a = Vector{matrix_[periodic_idx_1]};
            auto b = Vector{matrix_[periodic_idx_2]};
            auto c = a.cross(b).normalize();

            // Assign back to the matrix picking the "non-periodic" index without ifs
            matrix_[(3 - periodic_idx_1 - periodic_idx_2)] = c;
        }

        // precompute the inverse matrix
        auto det = matrix_.determinant();
        if (std::abs(det) < 1e-30) {
            throw std::runtime_error("the box matrix is not invertible");
        }

        this->inverse_ = matrix_.inverse();
    }

    const Matrix& matrix() const {
        return this->matrix_;
    }

    bool periodic(size_t spatial) const {
        return this->periodic_[spatial];
    }

    /// Convert a vector from cartesian coordinates to fractional coordinates
    Vector cartesian_to_fractional(Vector cartesian) const {
        return cartesian * inverse_;
    }

    /// Convert a vector from fractional coordinates to cartesian coordinates
    Vector fractional_to_cartesian(Vector fractional) const {
        return fractional * matrix_;
    }

    /// Get the three distances between faces of the bounding box
    Vector distances_between_faces() const {
        auto a = Vector{matrix_[0]};
        auto b = Vector{matrix_[1]};
        auto c = Vector{matrix_[2]};

        // Plans normal vectors
        auto na = b.cross(c).normalize();
        auto nb = c.cross(a).normalize();
        auto nc = a.cross(b).normalize();

        return Vector{
            std::abs(na.dot(a)),
            std::abs(nb.dot(b)),
            std::abs(nc.dot(c)),
        };
    }

private:
    Matrix matrix_;
    Matrix inverse_;
    std::array<bool, 3> periodic_;
};

/// A cell shift represents the displacement along cell axis between the actual
/// position of an atom and a periodic image of this atom.
///
/// The cell shift can be used to reconstruct the vector between two points,
/// wrapped inside the unit cell.
struct CellShift: public std::array<int32_t, 3> {
    /// Compute the shift vector in cartesian coordinates, using the given cell
    /// matrix (stored in row major order).
    Vector cartesian(Matrix cell) const {
        auto vector = Vector{
            static_cast<double>((*this)[0]),
            static_cast<double>((*this)[1]),
            static_cast<double>((*this)[2]),
        };
        return vector * cell;
    }
};

inline CellShift operator+(CellShift a, CellShift b) {
    return CellShift{
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
    };
}

inline CellShift operator-(CellShift a, CellShift b) {
    return CellShift{
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    };
}

} // namespace vesin

#endif
