#ifndef VESIN_MATH_HPP
#define VESIN_MATH_HPP

#include <array>
#include <cmath>
#include <stdexcept>

namespace vesin {
struct Vector;

Vector operator*(Vector vector, double scalar);

struct Vector: public std::array<double, 3> {
    double dot(Vector other) const {
        return (*this)[0] * other[0] + (*this)[1] * other[1] + (*this)[2] * other[2];
    }

    double norm() const {
        return std::sqrt(this->dot(*this));
    }

    Vector normalize() const {
        return *this * (1.0 / this->norm());
    }

    Vector cross(Vector other) const {
        return Vector{
            (*this)[1] * other[2] - (*this)[2] * other[1],
            (*this)[2] * other[0] - (*this)[0] * other[2],
            (*this)[0] * other[1] - (*this)[1] * other[0],
        };
    }
};

inline Vector operator+(Vector u, Vector v) {
    return Vector{
        u[0] + v[0],
        u[1] + v[1],
        u[2] + v[2],
    };
}

inline Vector operator-(Vector u, Vector v) {
    return Vector{
        u[0] - v[0],
        u[1] - v[1],
        u[2] - v[2],
    };
}

inline Vector operator*(double scalar, Vector vector) {
    return Vector{
        scalar * vector[0],
        scalar * vector[1],
        scalar * vector[2],
    };
}

inline Vector operator*(Vector vector, double scalar) {
    return Vector{
        scalar * vector[0],
        scalar * vector[1],
        scalar * vector[2],
    };
}

struct Matrix: public std::array<std::array<double, 3>, 3> {
    double determinant() const {
        // clang-format off
        return (*this)[0][0] * ((*this)[1][1] * (*this)[2][2] - (*this)[2][1] * (*this)[1][2])
             - (*this)[0][1] * ((*this)[1][0] * (*this)[2][2] - (*this)[1][2] * (*this)[2][0])
             + (*this)[0][2] * ((*this)[1][0] * (*this)[2][1] - (*this)[1][1] * (*this)[2][0]);
        // clang-format on
    }

    Matrix inverse() const {
        auto det = this->determinant();

        if (std::abs(det) < 1e-30) {
            throw std::runtime_error("this matrix is not invertible");
        }

        auto inverse = Matrix();
        inverse[0][0] = ((*this)[1][1] * (*this)[2][2] - (*this)[2][1] * (*this)[1][2]) / det;
        inverse[0][1] = ((*this)[0][2] * (*this)[2][1] - (*this)[0][1] * (*this)[2][2]) / det;
        inverse[0][2] = ((*this)[0][1] * (*this)[1][2] - (*this)[0][2] * (*this)[1][1]) / det;
        inverse[1][0] = ((*this)[1][2] * (*this)[2][0] - (*this)[1][0] * (*this)[2][2]) / det;
        inverse[1][1] = ((*this)[0][0] * (*this)[2][2] - (*this)[0][2] * (*this)[2][0]) / det;
        inverse[1][2] = ((*this)[1][0] * (*this)[0][2] - (*this)[0][0] * (*this)[1][2]) / det;
        inverse[2][0] = ((*this)[1][0] * (*this)[2][1] - (*this)[2][0] * (*this)[1][1]) / det;
        inverse[2][1] = ((*this)[2][0] * (*this)[0][1] - (*this)[0][0] * (*this)[2][1]) / det;
        inverse[2][2] = ((*this)[0][0] * (*this)[1][1] - (*this)[1][0] * (*this)[0][1]) / det;
        return inverse;
    }
};

inline Vector operator*(Matrix matrix, Vector vector) {
    return Vector{
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    };
}

inline Vector operator*(Vector vector, Matrix matrix) {
    return Vector{
        vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0],
        vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1],
        vector[0] * matrix[0][2] + vector[1] * matrix[1][2] + vector[2] * matrix[2][2],
    };
}

} // namespace vesin

#endif
