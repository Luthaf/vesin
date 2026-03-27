#ifndef VESIN_TYPES_HPP
#define VESIN_TYPES_HPP

#include <array>
#include <cassert>
#include <string>

#include "math.hpp"

namespace vesin {

class BoundingBox {
public:
    BoundingBox(const BoundingBox&) = delete;
    BoundingBox& operator=(const BoundingBox&) = delete;

    BoundingBox(BoundingBox&&) = default;
    BoundingBox& operator=(BoundingBox&&) = default;

    BoundingBox(Matrix matrix, const bool periodic[3]):
        matrix_(matrix),
        periodic_({periodic[0], periodic[1], periodic[2]}),
        max_positions_({-1e300, -1e300, -1e300}),
        min_positions_({1e300, 1e300, 1e300}) {

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

        // precompute distances between faces of the bounding box
        auto a = Vector{matrix_[0]};
        auto b = Vector{matrix_[1]};
        auto c = Vector{matrix_[2]};

        // Plans normal vectors
        auto na = b.cross(c).normalize();
        auto nb = c.cross(a).normalize();
        auto nc = a.cross(b).normalize();

        distances_between_faces_ = Vector{
            periodic_[0] ? std::abs(na.dot(a)) : max_positions_[0] - min_positions_[0],
            periodic_[1] ? std::abs(nb.dot(b)) : max_positions_[1] - min_positions_[1],
            periodic_[2] ? std::abs(nc.dot(c)) : max_positions_[2] - min_positions_[2],
        };
    }

    const Matrix& matrix() const {
        return this->matrix_;
    }

    bool periodic(size_t spatial) const {
        return this->periodic_[spatial];
    }

    /// Convert a vector from cartesian coordinates to fractional coordinates
    ///
    /// For non-periodic dimensions, the fractional coordinates are not wrapped
    /// inside [0, 1], but are normalized by the corresponding box length.
    Vector cartesian_to_fractional(Vector cartesian) const {
        auto fractional = cartesian * inverse_;
        if (!periodic_[0]) {
            fractional[0] = (cartesian[0] - min_positions_[0]) / distances_between_faces_[0];
        }

        if (!periodic_[1]) {
            fractional[1] = (cartesian[1] - min_positions_[1]) / distances_between_faces_[1];
        }

        if (!periodic_[2]) {
            fractional[2] = (cartesian[2] - min_positions_[2]) / distances_between_faces_[2];
        }

        return fractional;
    }

    /// Convert a vector from fractional coordinates to cartesian coordinates
    Vector fractional_to_cartesian(Vector fractional) const {
        auto cartesian = fractional * matrix_;

        if (!periodic_[0]) {
            cartesian[0] *= distances_between_faces_[0];
            cartesian[0] += min_positions_[0];
        }

        if (!periodic_[1]) {
            cartesian[1] *= distances_between_faces_[1];
            cartesian[1] += min_positions_[1];
        }

        if (!periodic_[2]) {
            cartesian[2] *= distances_between_faces_[2];
            cartesian[2] += min_positions_[2];
        }

        return cartesian;
    }

    /// Get the three distances between faces of the bounding box
    Vector distances_between_faces() const {
        return distances_between_faces_;
    }

    void make_bounding_for(const double (*points)[3], size_t n_points) {
        // find the min and max coordinates along each axis
        for (size_t i = 0; i < n_points; i++) {
            for (size_t spatial = 0; spatial < 3; spatial++) {
                if (!std::isfinite(points[i][spatial])) {
                    throw std::runtime_error(
                        "point " + std::to_string(i) + " has non-finite coordinate " +
                        "along axis " + std::to_string(spatial) + ": " +
                        std::to_string(points[i][spatial])
                    );
                }

                if (points[i][spatial] < min_positions_[spatial]) {
                    min_positions_[spatial] = points[i][spatial];
                }
                if (points[i][spatial] > max_positions_[spatial]) {
                    max_positions_[spatial] = points[i][spatial];
                }
            }
        }

        for (int dim = 0; dim < 3; dim++) {
            // if all atoms have the same coordinate in this dimension, pretend
            // that the bounding box is at least 1 unit wide to avoid numerical issues
            if (max_positions_[dim] - min_positions_[dim] < 1e-6) {
                max_positions_[dim] = min_positions_[dim] + 1;
            }

            if (!periodic_[dim]) {
                // add a 1% margin to make sure all points are strictly inside the
                // bounding box
                distances_between_faces_[dim] = max_positions_[dim] * 1.01 - min_positions_[dim];
            }
        }
    }

private:
    Matrix matrix_;
    std::array<bool, 3> periodic_;

    Matrix inverse_;
    Vector min_positions_;
    Vector max_positions_;
    Vector distances_between_faces_;
};

/// A cell shift represents the displacement along cell axis between the actual
/// position of an atom and a periodic image of this atom.
///
/// The cell shift can be used to reconstruct the vector between two points,
/// wrapped inside the unit cell.
struct CellShift: public std::array<int32_t, 3> {
    /// Compute the shift vector in cartesian coordinates, using the given cell
    /// matrix (stored in row major order).
    Vector cartesian(const BoundingBox& box) const {
        assert(box.periodic(0) || (*this)[0] == 0);
        assert(box.periodic(1) || (*this)[1] == 0);
        assert(box.periodic(2) || (*this)[2] == 0);

        auto vector = Vector{
            static_cast<double>((*this)[0]),
            static_cast<double>((*this)[1]),
            static_cast<double>((*this)[2]),
        };
        return vector * box.matrix();
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
