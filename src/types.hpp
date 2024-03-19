#ifndef VESIN_TYPES_HPP
#define VESIN_TYPES_HPP

#include "math.hpp"

namespace vesin {

class BoundingBox {
public:
    BoundingBox(Matrix matrix, bool periodic): matrix_(std::move(matrix)), periodic_(periodic) {
        if (periodic) {
            this->inverse_ = matrix_.inverse();
        } else {
            this->matrix_ = Matrix{{{
                {{1, 0, 0}},
                {{0, 1, 0}},
                {{0, 0, 1}}
            }}};
            this->inverse_ = matrix_;
        }
    }

    const Matrix& matrix() const {
        return this->matrix_;
    }

    bool periodic() const {
        return this->periodic_;
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
    bool periodic_;
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



}

#endif
