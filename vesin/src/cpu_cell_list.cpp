#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <new>
#include <numeric>
#include <tuple>

#include "cpu_cell_list.hpp"

using namespace vesin::cpu;

void vesin::cpu::neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox cell,
    VesinOptions options,
    VesinNeighborList& raw_neighbors
) {
    auto cell_list = CellList(cell, options.cutoff);

    for (size_t i = 0; i < n_points; i++) {
        cell_list.add_point(i, points[i]);
    }

    auto cell_matrix = cell.matrix();
    auto cutoff2 = options.cutoff * options.cutoff;

    // the cell list creates too many pairs, we only need to keep the
    // one where the distance is actually below the cutoff
    auto neighbors = GrowableNeighborList{raw_neighbors, raw_neighbors.length, options};
    neighbors.reset();

    cell_list.foreach_pair([&](size_t first, size_t second, CellShift shift) {
        if (!options.full) {
            // filter out some pairs for half neighbor lists
            if (first > second) {
                return;
            }

            if (first == second) {
                // When creating pairs between a point and one of its periodic
                // images, the code generate multiple redundant pairs (e.g. with
                // shifts 0 1 1 and 0 -1 -1); and we want to only keep one of
                // these.
                if (shift[0] + shift[1] + shift[2] < 0) {
                    // drop shifts on the negative half-space
                    return;
                }

                if ((shift[0] + shift[1] + shift[2] == 0) && (shift[2] < 0 || (shift[2] == 0 && shift[1] < 0))) {
                    // drop shifts in the negative half plane or the negative
                    // shift[1] axis. See below for a graphical representation:
                    // we are keeping the shifts indicated with `O` and dropping
                    // the ones indicated with `X`
                    //
                    //  O O O │ O O O
                    //  O O O │ O O O
                    //  O O O │ O O O
                    // ─X─X─X─┼─O─O─O─
                    //  X X X │ X X X
                    //  X X X │ X X X
                    //  X X X │ X X X
                    return;
                }
            }
        }

        auto vector = points[second] - points[first] + shift.cartesian(cell_matrix);
        auto distance2 = vector.dot(vector);

        if (distance2 < cutoff2) {
            auto index = neighbors.length();
            neighbors.set_pair(index, first, second);

            if (options.return_shifts) {
                neighbors.set_shift(index, shift);
            }

            if (options.return_distances) {
                neighbors.set_distance(index, std::sqrt(distance2));
            }

            if (options.return_vectors) {
                neighbors.set_vector(index, vector);
            }

            neighbors.increment_length();
        }
    });

    if (options.sorted) {
        neighbors.sort();
    }
}

/* ========================================================================== */

/// Maximal number of cells, we need to use this to prevent having too many
/// cells with a small bounding box and a large cutoff
#define MAX_NUMBER_OF_CELLS 1e5

/// Function to compute both quotient and remainder of the division of a by b.
/// This function follows Python convention, making sure the remainder have the
/// same sign as `b`.
static std::tuple<int32_t, int32_t> divmod(int32_t a, size_t b) {
    assert(b < (std::numeric_limits<int32_t>::max()));
    auto b_32 = static_cast<int32_t>(b);
    auto quotient = a / b_32;
    auto remainder = a % b_32;
    if (remainder < 0) {
        remainder += b_32;
        quotient -= 1;
    }
    return std::make_tuple(quotient, remainder);
}

/// Apply the `divmod` function to three components at the time
static std::tuple<std::array<int32_t, 3>, std::array<int32_t, 3>>
divmod(std::array<int32_t, 3> a, std::array<size_t, 3> b) {
    auto [qx, rx] = divmod(a[0], b[0]);
    auto [qy, ry] = divmod(a[1], b[1]);
    auto [qz, rz] = divmod(a[2], b[2]);
    return std::make_tuple(
        std::array<int32_t, 3>{qx, qy, qz},
        std::array<int32_t, 3>{rx, ry, rz}
    );
}

CellList::CellList(BoundingBox box, double cutoff):
    n_search_({0, 0, 0}),
    cells_shape_({0, 0, 0}),
    box_(box) {
    auto distances_between_faces = box_.distances_between_faces();

    auto n_cells = Vector{
        std::clamp(std::trunc(distances_between_faces[0] / cutoff), 1.0, HUGE_VAL),
        std::clamp(std::trunc(distances_between_faces[1] / cutoff), 1.0, HUGE_VAL),
        std::clamp(std::trunc(distances_between_faces[2] / cutoff), 1.0, HUGE_VAL),
    };

    assert(std::isfinite(n_cells[0]) && std::isfinite(n_cells[1]) && std::isfinite(n_cells[2]));

    // limit memory consumption by ensuring we have less than `MAX_N_CELLS`
    // cells to look though
    auto n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
    if (n_cells_total > MAX_NUMBER_OF_CELLS) {
        // set the total number of cells close to MAX_N_CELLS, while keeping
        // roughly the ratio of cells in each direction
        auto ratio_x_y = n_cells[0] / n_cells[1];
        auto ratio_y_z = n_cells[1] / n_cells[2];

        n_cells[2] = std::trunc(std::cbrt(MAX_NUMBER_OF_CELLS / (ratio_x_y * ratio_y_z * ratio_y_z)));
        n_cells[1] = std::trunc(ratio_y_z * n_cells[2]);
        n_cells[0] = std::trunc(ratio_x_y * n_cells[1]);
    }

    // number of cells to search in each direction to make sure all possible
    // pairs below the cutoff are accounted for.
    this->n_search_ = std::array<int32_t, 3>{
        static_cast<int32_t>(std::ceil(cutoff * n_cells[0] / distances_between_faces[0])),
        static_cast<int32_t>(std::ceil(cutoff * n_cells[1] / distances_between_faces[1])),
        static_cast<int32_t>(std::ceil(cutoff * n_cells[2] / distances_between_faces[2])),
    };

    this->cells_shape_ = std::array<size_t, 3>{
        static_cast<size_t>(n_cells[0]),
        static_cast<size_t>(n_cells[1]),
        static_cast<size_t>(n_cells[2]),
    };

    for (size_t spatial = 0; spatial < 3; spatial++) {
        if (n_search_[spatial] < 1) {
            n_search_[spatial] = 1;
        }

        // don't look for neighboring cells if we have only one cell and no
        // periodic boundary condition (for this spatial direction)
        if (n_cells[spatial] == 1 && !box.periodic()[spatial]) {
            n_search_[spatial] = 0;
        }
    }

    this->cells_.resize(cells_shape_[0] * cells_shape_[1] * cells_shape_[2]);
}

void CellList::add_point(size_t index, Vector position) {
    auto fractional = box_.cartesian_to_fractional(position);

    // find the cell in which this atom should go
    auto cell_index = std::array<int32_t, 3>{
        static_cast<int32_t>(std::floor(fractional[0] * static_cast<double>(cells_shape_[0]))),
        static_cast<int32_t>(std::floor(fractional[1] * static_cast<double>(cells_shape_[1]))),
        static_cast<int32_t>(std::floor(fractional[2] * static_cast<double>(cells_shape_[2]))),
    };

    // deal with pbc by wrapping the atom inside if it was outside of the cell
    CellShift shift{{0, 0, 0}};
    for (size_t axis = 0; axis < 3; axis++) {
        if (box_.periodic(axis)) {
            auto result = divmod(cell_index[axis], cells_shape_[axis]);
            shift[axis] = std::get<0>(result);
            cell_index[axis] = std::get<1>(result);
        } else {
            shift[axis] = 0;
            cell_index[axis] = std::clamp(
                cell_index[axis],
                0,
                static_cast<int32_t>(cells_shape_[axis] - 1)
            );
        }
    }
    this->get_cell(cell_index).emplace_back(Point{index, shift});
}

// clang-format off
template <typename Function>
void CellList::foreach_pair(Function callback) {
    for (int32_t cell_i_x=0; cell_i_x<static_cast<int32_t>(cells_shape_[0]); cell_i_x++) {
    for (int32_t cell_i_y=0; cell_i_y<static_cast<int32_t>(cells_shape_[1]); cell_i_y++) {
    for (int32_t cell_i_z=0; cell_i_z<static_cast<int32_t>(cells_shape_[2]); cell_i_z++) {
        const auto& current_cell = this->get_cell({cell_i_x, cell_i_y, cell_i_z});
        // look through each neighboring cell
        for (int32_t delta_x=-n_search_[0]; delta_x<=n_search_[0]; delta_x++) {
        for (int32_t delta_y=-n_search_[1]; delta_y<=n_search_[1]; delta_y++) {
        for (int32_t delta_z=-n_search_[2]; delta_z<=n_search_[2]; delta_z++) {
            auto cell_i = std::array<int32_t, 3>{
                cell_i_x + delta_x,
                cell_i_y + delta_y,
                cell_i_z + delta_z,
            };

            // shift vector from one cell to the other and index of
            // the neighboring cell
            auto neighbor_cell_i = cell_i;
            CellShift cell_shift({0, 0, 0});

            bool is_neighboring_cell_outside_bounds = calc_neighbor_cell_shifts_and_check_outside_bounds(neighbor_cell_i, cell_shift);
            if (is_neighboring_cell_outside_bounds) {
                continue;
            }

            for (const auto& atom_i: current_cell) {
                for (const auto& atom_j: this->get_cell(neighbor_cell_i)) {
                    auto shift = CellShift{cell_shift} + atom_i.shift - atom_j.shift;
                    auto shift_is_zero = shift[0] == 0 && shift[1] == 0 && shift[2] == 0;

                    if (atom_i.index == atom_j.index && shift_is_zero) {
                        // only create pairs with the same atom twice if the
                        // pair spans more than one bounding box
                        continue;
                    }

                    callback(atom_i.index, atom_j.index, shift);
                }
            } // loop over atoms in current neighbor cells
        }}}
    }}} // loop over neighboring cells
}

bool CellList::calc_neighbor_cell_shifts_and_check_outside_bounds(std::array<int32_t, 3>& neighbor_cell_i, CellShift& cell_shift) const {
    for (size_t axis = 0; axis < 3; axis++) {
        if (box_.periodic(axis)) {
            // since we have periodic boundary conditions for this axis, there will always be a neighboring cell
            // we use divmod to find the shift and the neighboring cell index
            auto [quotient, remainder] = divmod(neighbor_cell_i[axis], cells_shape_[axis]);
            cell_shift[axis] = quotient;
            neighbor_cell_i[axis] = remainder;
        } else {
            // if periodicity is disabled for this axis, we skip this cell if it is outside the simulation bounds
            if (neighbor_cell_i[axis] < 0 || neighbor_cell_i[axis] >= static_cast<int32_t>(cells_shape_[axis])) {
                return true;
            }
        }
    }
    return false;
}

CellList::Cell& CellList::get_cell(std::array<int32_t, 3> index) {
    size_t linear_index = (cells_shape_[0] * cells_shape_[1] * index[2])
                        + (cells_shape_[0] * index[1])
                        + index[0];
    return cells_[linear_index];
}
// clang-format on

/* ========================================================================== */

void GrowableNeighborList::set_pair(size_t index, size_t first, size_t second) {
    if (index >= this->capacity) {
        this->grow();
    }

    this->neighbors.pairs[index][0] = first;
    this->neighbors.pairs[index][1] = second;
}

void GrowableNeighborList::set_shift(size_t index, vesin::CellShift shift) {
    if (index >= this->capacity) {
        this->grow();
    }

    this->neighbors.shifts[index][0] = shift[0];
    this->neighbors.shifts[index][1] = shift[1];
    this->neighbors.shifts[index][2] = shift[2];
}

void GrowableNeighborList::set_distance(size_t index, double distance) {
    if (index >= this->capacity) {
        this->grow();
    }

    this->neighbors.distances[index] = distance;
}

void GrowableNeighborList::set_vector(size_t index, vesin::Vector vector) {
    if (index >= this->capacity) {
        this->grow();
    }

    this->neighbors.vectors[index][0] = vector[0];
    this->neighbors.vectors[index][1] = vector[1];
    this->neighbors.vectors[index][2] = vector[2];
}

template <typename scalar_t, size_t N>
static scalar_t (*alloc(scalar_t (*ptr)[N], size_t size, size_t new_size))[N] {
    auto* new_ptr = reinterpret_cast<scalar_t(*)[N]>(std::realloc(ptr, new_size * sizeof(scalar_t[N])));

    if (new_ptr == nullptr) {
        throw std::bad_alloc();
    }

    // initialize with a bit pattern that maps to NaN for double
    std::memset(new_ptr + size, 0b11111111, (new_size - size) * sizeof(scalar_t[N]));

    return new_ptr;
}

template <typename scalar_t>
static scalar_t* alloc(scalar_t* ptr, size_t size, size_t new_size) {
    auto* new_ptr = reinterpret_cast<scalar_t*>(std::realloc(ptr, new_size * sizeof(scalar_t)));

    if (new_ptr == nullptr) {
        throw std::bad_alloc();
    }

    // initialize with a bit pattern that maps to NaN for double
    std::memset(new_ptr + size, 0b11111111, (new_size - size) * sizeof(scalar_t));

    return new_ptr;
}

void GrowableNeighborList::grow() {
    auto new_size = neighbors.length * 2;
    if (new_size == 0) {
        new_size = 1;
    }

    auto* new_pairs = alloc<size_t, 2>(neighbors.pairs, neighbors.length, new_size);

    int32_t (*new_shifts)[3] = nullptr;
    if (options.return_shifts) {
        new_shifts = alloc<int32_t, 3>(neighbors.shifts, neighbors.length, new_size);
    }

    double* new_distances = nullptr;
    if (options.return_distances) {
        new_distances = alloc<double>(neighbors.distances, neighbors.length, new_size);
    }

    double (*new_vectors)[3] = nullptr;
    if (options.return_vectors) {
        new_vectors = alloc<double, 3>(neighbors.vectors, neighbors.length, new_size);
    }

    this->neighbors.pairs = new_pairs;
    this->neighbors.shifts = new_shifts;
    this->neighbors.distances = new_distances;
    this->neighbors.vectors = new_vectors;

    this->capacity = new_size;
}

void GrowableNeighborList::reset() {
    // set all allocated data to zero
    auto size = this->neighbors.length;
    std::memset(this->neighbors.pairs, 0, size * sizeof(size_t[2]));

    if (this->neighbors.shifts != nullptr) {
        std::memset(this->neighbors.shifts, 0, size * sizeof(int32_t[3]));
    }

    if (this->neighbors.distances != nullptr) {
        std::memset(this->neighbors.distances, 0, size * sizeof(double));
    }

    if (this->neighbors.vectors != nullptr) {
        std::memset(this->neighbors.vectors, 0, size * sizeof(double[3]));
    }

    // reset length (but keep the capacity where it's at)
    this->neighbors.length = 0;

    // allocate/deallocate pointers as required
    auto* shifts = this->neighbors.shifts;
    if (this->options.return_shifts && shifts == nullptr) {
        shifts = alloc<int32_t, 3>(shifts, 0, capacity);
    } else if (!this->options.return_shifts && shifts != nullptr) {
        std::free(shifts);
        shifts = nullptr;
    }

    auto* distances = this->neighbors.distances;
    if (this->options.return_distances && distances == nullptr) {
        distances = alloc<double>(distances, 0, capacity);
    } else if (!this->options.return_distances && distances != nullptr) {
        std::free(distances);
        distances = nullptr;
    }

    auto* vectors = this->neighbors.vectors;
    if (this->options.return_vectors && vectors == nullptr) {
        vectors = alloc<double, 3>(vectors, 0, capacity);
    } else if (!this->options.return_vectors && vectors != nullptr) {
        std::free(vectors);
        vectors = nullptr;
    }

    this->neighbors.shifts = shifts;
    this->neighbors.distances = distances;
    this->neighbors.vectors = vectors;
}

void GrowableNeighborList::sort() {
    if (this->length() == 0) {
        return;
    }

    // step 1: sort an array of indices, comparing the pairs at the indices
    auto indices = std::vector<int64_t>(this->length(), 0);
    std::iota(std::begin(indices), std::end(indices), 0);

    struct compare_pairs {
        compare_pairs(size_t (*pairs_)[2]):
            pairs(pairs_) {}

        bool operator()(int64_t a, int64_t b) const {
            if (pairs[a][0] == pairs[b][0]) {
                return pairs[a][1] < pairs[b][1];
            } else {
                return pairs[a][0] < pairs[b][0];
            }
        }

        size_t (*pairs)[2];
    };

    std::sort(std::begin(indices), std::end(indices), compare_pairs(this->neighbors.pairs));

    // step 2: permute all data according to the sorted indices.
    int64_t cur = 0;
    int64_t is_sorted_up_to = 0;
    // data in `from` should go to `cur`
    auto from = indices[cur];

    size_t tmp_pair[2] = {0};
    double tmp_distance = 0;
    double tmp_vector[3] = {0};
    int32_t tmp_shift[3] = {0};

    while (cur < this->length()) {
        // move data from `cur` to temporary
        std::swap(tmp_pair, this->neighbors.pairs[cur]);
        if (options.return_distances) {
            std::swap(tmp_distance, this->neighbors.distances[cur]);
        }
        if (options.return_vectors) {
            std::swap(tmp_vector, this->neighbors.vectors[cur]);
        }
        if (options.return_shifts) {
            std::swap(tmp_shift, this->neighbors.shifts[cur]);
        }

        from = indices[cur];
        do {
            if (from == cur) {
                // permutation loop of a single entry, i.e. this value stayed
                // where is already was
                break;
            }
            // move data from `from` to `cur`
            std::swap(this->neighbors.pairs[cur], this->neighbors.pairs[from]);
            if (options.return_distances) {
                std::swap(this->neighbors.distances[cur], this->neighbors.distances[from]);
            }
            if (options.return_vectors) {
                std::swap(this->neighbors.vectors[cur], this->neighbors.vectors[from]);
            }
            if (options.return_shifts) {
                std::swap(this->neighbors.shifts[cur], this->neighbors.shifts[from]);
            }

            // mark this spot as already visited
            indices[cur] = -1;

            // update the indices
            cur = from;
            from = indices[cur];
        } while (indices[from] != -1);

        // we found a full loop of permutation, we can put tmp into `cur`
        std::swap(this->neighbors.pairs[cur], tmp_pair);
        if (options.return_distances) {
            std::swap(this->neighbors.distances[cur], tmp_distance);
        }
        if (options.return_vectors) {
            std::swap(this->neighbors.vectors[cur], tmp_vector);
        }
        if (options.return_shifts) {
            std::swap(this->neighbors.shifts[cur], tmp_shift);
        }

        indices[cur] = -1;

        // look for the next loop of permutation
        cur = is_sorted_up_to;
        while (indices[cur] == -1) {
            cur += 1;
            is_sorted_up_to += 1;
            if (cur == this->length()) {
                break;
            }
        }
    }
}

void vesin::cpu::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCPU);

    std::free(neighbors.pairs);
    std::free(neighbors.shifts);
    std::free(neighbors.vectors);
    std::free(neighbors.distances);
}
