#include <cassert>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <type_traits>

#include "cpu_cell_list.hpp"
#include "threadpool.hpp"
#include "verlet.hpp"

using namespace vesin::cpu;

static void free_extra_data(VesinNeighborList& neighbors) {
    delete static_cast<ExtraDataCpu*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

static ExtraDataCpu& extra_data(VesinNeighborList& neighbors) {
    if (neighbors.opaque == nullptr) {
        neighbors.opaque = new ExtraDataCpu();
    }

    return *static_cast<ExtraDataCpu*>(neighbors.opaque);
}

static void validate_algorithm(VesinOptions options) {
    if (options.algorithm != VesinAutoAlgorithm && options.algorithm != VesinCellList) {
        throw std::runtime_error("only VesinAutoAlgorithm and VesinCellList are supported on CPU");
    }
}

ExtraDataCpu::~ExtraDataCpu() {
    delete this->verlet_state;
}

struct ThreadLocalNeighborLists {
    ThreadLocalNeighborLists(size_t n_threads):
        raw(n_threads - 1) {
        for (auto& list : raw) {
            list.device = {VesinCPU, 0};
        }
    }

    std::vector<VesinNeighborList> raw;

    ~ThreadLocalNeighborLists() {
        for (auto& list : raw) {
            vesin::cpu::free_neighbors(list);
        }
    }
};

static ThreadPool& global_thread_pool() {
    static auto pool = ThreadPool();
    return pool;
}

void vesin::cpu::cell_list_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& raw_neighbors,
    size_t& capacity
) {
    validate_algorithm(options);

    auto cell_list = CellList(std::move(box), options.cutoff);

    for (size_t i = 0; i < n_points; i++) {
        cell_list.add_point(i, points[i]);
    }

    auto cutoff2 = options.cutoff * options.cutoff;

    // the cell list creates too many pairs, we only need to keep the
    // one where the distance is actually below the cutoff
    auto initial_capacity = std::max(capacity, raw_neighbors.length);
    auto neighbors = GrowableNeighborList{raw_neighbors, initial_capacity, options};
    neighbors.reset();

    auto thread_count = std::max<size_t>(1, static_cast<size_t>(options.n_threads));
    thread_count = std::min(thread_count, cell_list.n_cells());
    auto& thread_pool = global_thread_pool();

    auto thread_locals = ThreadLocalNeighborLists(thread_count);

    auto local_thread_neighbors = std::vector<GrowableNeighborList>();
    local_thread_neighbors.reserve(thread_locals.raw.size());
    for (auto& local_raw : thread_locals.raw) {
        local_raw.device = {VesinCPU, 0};
        auto list = GrowableNeighborList{local_raw, 0, options};
        list.reset();
        local_thread_neighbors.emplace_back(list);
    }

    cell_list.foreach_pair(thread_pool, thread_count, [&](size_t first, size_t second, CellShift shift, size_t thread_id) {
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

        auto vector = points[second] - points[first] + shift.cartesian(box);
        auto distance2 = vector.dot(vector);

        if (distance2 < cutoff2) {
            GrowableNeighborList* local_list = nullptr;
            if (thread_id == 0) {
                local_list = &neighbors;
            } else {
                local_list = &local_thread_neighbors[thread_id - 1];
            }

            auto index = local_list->length();
            local_list->set_pair(index, first, second);

            if (options.return_shifts) {
                local_list->set_shift(index, shift);
            }

            if (options.return_distances) {
                local_list->set_distance(index, std::sqrt(distance2));
            }

            if (options.return_vectors) {
                local_list->set_vector(index, vector);
            }

            local_list->increment_length();
        }
    });

    // Threads 1 to n-1 have their neighbors in `thread_locals`; merge them
    // into the list created by thread 0 (`raw_neighbors`).
    auto final_length = neighbors.length();
    auto local_offsets = std::vector<size_t>(thread_locals.raw.size(), 0);
    for (size_t local_i = 0; local_i < thread_locals.raw.size(); local_i++) {
        local_offsets[local_i] = final_length;
        final_length += thread_locals.raw[local_i].length;
    }

    neighbors.grow(final_length);

    thread_pool.run(thread_count, thread_locals.raw.size(), [&](size_t local_i, size_t /*thread_id*/) {
        const auto& local_neighbors = thread_locals.raw[local_i];
        auto offset = local_offsets[local_i];

        for (size_t i = 0; i < local_neighbors.length; i++) {
            auto index = offset + i;
            neighbors.neighbors.pairs[index][0] = local_neighbors.pairs[i][0];
            neighbors.neighbors.pairs[index][1] = local_neighbors.pairs[i][1];

            if (options.return_shifts) {
                neighbors.neighbors.shifts[index][0] = local_neighbors.shifts[i][0];
                neighbors.neighbors.shifts[index][1] = local_neighbors.shifts[i][1];
                neighbors.neighbors.shifts[index][2] = local_neighbors.shifts[i][2];
            }

            if (options.return_distances) {
                neighbors.neighbors.distances[index] = local_neighbors.distances[i];
            }

            if (options.return_vectors) {
                neighbors.neighbors.vectors[index][0] = local_neighbors.vectors[i][0];
                neighbors.neighbors.vectors[index][1] = local_neighbors.vectors[i][1];
                neighbors.neighbors.vectors[index][2] = local_neighbors.vectors[i][2];
            }
        }
    });

    neighbors.neighbors.length = final_length;

    if (options.sorted) {
        neighbors.sort();
    }

    capacity = neighbors.capacity;
}

void vesin::cpu::neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    VesinNeighborList& raw_neighbors
) {
    validate_algorithm(options);

    auto& extra = extra_data(raw_neighbors);
    if (options.skin > 0.0) {
        if (extra.verlet_state == nullptr) {
            extra.verlet_state = new VerletList();
        }

        auto& state = *extra.verlet_state;
        state.set_options(options);

        if (state.needs_rebuild(points, n_points, box)) {
            state.rebuild(points, n_points, box);
        }

        state.filter(points, box, options, raw_neighbors, extra.capacity);
    } else {
        // Remove verlet state if it exists from a previous call
        delete extra.verlet_state;
        extra.verlet_state = nullptr;

        cell_list_neighbors(points, n_points, std::move(box), options, raw_neighbors, extra.capacity);
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
    box_(std::move(box)) {
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
    }

    // When there is mixed periodicity (some dimensions periodic, some not),
    // periodic shifts can bring atoms together across multiple cells in
    // non-periodic dimensions. Increase the search range in non-periodic
    // dimensions to cover the full cell extent in those dimensions.
    bool has_periodic = false;
    bool has_non_periodic = false;
    for (size_t d = 0; d < 3; d++) {
        if (box_.periodic(d)) {
            has_periodic = true;
        } else {
            has_non_periodic = true;
        }
    }

    if (has_periodic && has_non_periodic) {
        for (size_t spatial = 0; spatial < 3; spatial++) {
            if (!box_.periodic(spatial)) {
                n_search_[spatial] = std::max(
                    n_search_[spatial],
                    static_cast<int32_t>(cells_shape_[spatial]) - 1
                );
            }
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
    CellShift shift;
    for (size_t spatial = 0; spatial < 3; spatial++) {
        auto result = divmod(cell_index[spatial], cells_shape_[spatial]);
        shift[spatial] = std::get<0>(result);
        cell_index[spatial] = std::get<1>(result);

        assert(box_.periodic(spatial) || shift[spatial] == 0);
    }

    this->get_cell(cell_index).emplace_back(Point{index, shift});
}

template <typename Function>
void CellList::foreach_pair(ThreadPool& thread_pool, size_t n_threads, Function callback) {
    auto total_cells = cells_shape_[0] * cells_shape_[1] * cells_shape_[2];

    thread_pool.run(n_threads, total_cells, [&](size_t linear_cell_i, size_t thread_id) {
        auto stride_x = cells_shape_[0];
        auto stride_xy = cells_shape_[0] * cells_shape_[1];

        auto cell_i_x = static_cast<int32_t>(linear_cell_i % stride_x);
        auto cell_i_y = static_cast<int32_t>((linear_cell_i / stride_x) % cells_shape_[1]);
        auto cell_i_z = static_cast<int32_t>(linear_cell_i / stride_xy);

        const auto& current_cell = this->get_cell({cell_i_x, cell_i_y, cell_i_z});

        // look through each neighboring cell
        for (int32_t delta_x = -n_search_[0]; delta_x <= n_search_[0]; delta_x++) {
            for (int32_t delta_y = -n_search_[1]; delta_y <= n_search_[1]; delta_y++) {
                for (int32_t delta_z = -n_search_[2]; delta_z <= n_search_[2]; delta_z++) {
                    // shift vector from one cell to the other
                    auto cell_shift = std::array<int32_t, 3>{0, 0, 0};
                    // index of the neighboring cell
                    auto neighbor_cell_i = std::array<int32_t, 3>{
                        cell_i_x + delta_x,
                        cell_i_y + delta_y,
                        cell_i_z + delta_z,
                    };

                    // only wrap (i.e. call divmod) cell indices in periodic dimensions,
                    // skip out-of-bounds cells in non-periodic ones
                    bool cell_is_valid = true;
                    for (int d = 0; d < 3; d++) {
                        if (box_.periodic(d)) {
                            auto [q, r] = divmod(neighbor_cell_i[d], cells_shape_[d]);
                            cell_shift[d] = q;
                            neighbor_cell_i[d] = r;
                        } else if (neighbor_cell_i[d] < 0 || neighbor_cell_i[d] >= static_cast<int32_t>(cells_shape_[d])) {
                            cell_is_valid = false;
                            break;
                        }
                    }

                    if (!cell_is_valid) {
                        continue;
                    }

                    for (const auto& atom_i : current_cell) {
                        for (const auto& atom_j : this->get_cell(neighbor_cell_i)) {
                            auto shift = CellShift(cell_shift) + atom_i.shift - atom_j.shift;
                            auto shift_is_zero = shift[0] == 0 && shift[1] == 0 && shift[2] == 0;

                            if ((shift[0] != 0 && !box_.periodic(0)) ||
                                (shift[1] != 0 && !box_.periodic(1)) ||
                                (shift[2] != 0 && !box_.periodic(2))) {
                                // do not create pairs crossing the periodic
                                // boundaries in a non-periodic box
                                continue;
                            }

                            if (atom_i.index == atom_j.index && shift_is_zero) {
                                // only create pairs with the same atom twice if the
                                // pair spans more than one bounding box
                                continue;
                            }

                            if constexpr (std::is_invocable_v<Function, size_t, size_t, CellShift, size_t>) {
                                callback(atom_i.index, atom_j.index, shift, thread_id);
                            } else {
                                callback(atom_i.index, atom_j.index, shift);
                            }
                        }
                    } // loop over atoms in current neighbor cells
                }
            }
        }
    });
}

CellList::Cell& CellList::get_cell(std::array<int32_t, 3> index) {
    size_t linear_index = (cells_shape_[0] * cells_shape_[1] * index[2]) + (cells_shape_[0] * index[1]) + index[0];
    return cells_[linear_index];
}

const CellList::Cell& CellList::get_cell(std::array<int32_t, 3> index) const {
    size_t linear_index = (cells_shape_[0] * cells_shape_[1] * index[2]) + (cells_shape_[0] * index[1]) + index[0];
    return cells_[linear_index];
}

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
using array_ptr = scalar_t (*)[N];

template <typename scalar_t, size_t N>
static array_ptr<scalar_t, N> alloc(array_ptr<scalar_t, N> ptr, size_t size, size_t new_size) {
    auto* new_ptr = reinterpret_cast<scalar_t(*)[N]>(std::realloc(ptr, new_size * sizeof(scalar_t[N])));

    if (new_ptr == nullptr) {
        return nullptr;
    }

#ifndef NDEBUG
    // initialize with a bit pattern that maps to NaN for double
    std::memset(new_ptr + size, 0b11111111, (new_size - size) * sizeof(scalar_t[N]));
#endif

    return new_ptr;
}

template <typename scalar_t>
static scalar_t* alloc(scalar_t* ptr, size_t size, size_t new_size) {
    auto* new_ptr = reinterpret_cast<scalar_t*>(std::realloc(ptr, new_size * sizeof(scalar_t)));

    if (new_ptr == nullptr) {
        return nullptr;
    }

#ifndef NDEBUG
    // initialize with a bit pattern that maps to NaN for double
    std::memset(new_ptr + size, 0b11111111, (new_size - size) * sizeof(scalar_t));
#endif

    return new_ptr;
}

void GrowableNeighborList::grow(size_t new_size) {
    assert(this->neighbors.device.type == VesinCPU);

    if (new_size == 0) {
        new_size = neighbors.length == 0 ? 1 : neighbors.length * 2;
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

    if (
        (new_pairs == nullptr) ||
        (options.return_shifts && new_shifts == nullptr) ||
        (options.return_distances && new_distances == nullptr) ||
        (options.return_vectors && new_vectors == nullptr)
    ) {
        std::free(new_pairs);
        std::free(new_shifts);
        std::free(new_distances);
        std::free(new_vectors);
        throw std::runtime_error("could not allocate memory for growing neighbor list");
    }

    this->neighbors.pairs = new_pairs;
    this->neighbors.shifts = new_shifts;
    this->neighbors.distances = new_distances;
    this->neighbors.vectors = new_vectors;

    this->capacity = new_size;
}

void GrowableNeighborList::reset() {
#ifndef NDEBUG
    auto size = this->neighbors.length;
    // set all allocated data to a bit pattern that maps to NaN for double
    std::memset(this->neighbors.pairs, 0b11111111, size * sizeof(size_t[2]));

    if (this->neighbors.shifts != nullptr) {
        std::memset(this->neighbors.shifts, 0b11111111, size * sizeof(int32_t[3]));
    }

    if (this->neighbors.distances != nullptr) {
        std::memset(this->neighbors.distances, 0b11111111, size * sizeof(double));
    }

    if (this->neighbors.vectors != nullptr) {
        std::memset(this->neighbors.vectors, 0b11111111, size * sizeof(double[3]));
    }
#endif

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
            return pairs[a][0] < pairs[b][0];
        }

        size_t (*pairs)[2];
    };

    std::sort(
        std::begin(indices),
        std::end(indices),
        compare_pairs(this->neighbors.pairs)
    );

    // step 2: move all data according to the sorted indices.
    auto* sorted_pairs = alloc<size_t, 2>(nullptr, 0, this->capacity);

    int32_t (*sorted_shifts)[3] = nullptr;
    if (options.return_shifts) {
        sorted_shifts = alloc<int32_t, 3>(nullptr, 0, this->capacity);
    }

    double* sorted_distances = nullptr;
    if (options.return_distances) {
        sorted_distances = alloc<double>(nullptr, 0, this->capacity);
    }

    double (*sorted_vectors)[3] = nullptr;
    if (options.return_vectors) {
        sorted_vectors = alloc<double, 3>(nullptr, 0, this->capacity);
    }

    if (
        (sorted_pairs == nullptr) ||
        (options.return_shifts && sorted_shifts == nullptr) ||
        (options.return_distances && sorted_distances == nullptr) ||
        (options.return_vectors && sorted_vectors == nullptr)
    ) {
        std::free(sorted_pairs);
        std::free(sorted_shifts);
        std::free(sorted_distances);
        std::free(sorted_vectors);
        throw std::runtime_error("could not allocate memory for sorting neighbor list");
    }

    for (size_t i = 0; i < this->neighbors.length; i++) {
        auto from = static_cast<size_t>(indices[i]);
        sorted_pairs[i][0] = this->neighbors.pairs[from][0];
        sorted_pairs[i][1] = this->neighbors.pairs[from][1];

        if (options.return_shifts) {
            sorted_shifts[i][0] = this->neighbors.shifts[from][0];
            sorted_shifts[i][1] = this->neighbors.shifts[from][1];
            sorted_shifts[i][2] = this->neighbors.shifts[from][2];
        }

        if (options.return_distances) {
            sorted_distances[i] = this->neighbors.distances[from];
        }

        if (options.return_vectors) {
            sorted_vectors[i][0] = this->neighbors.vectors[from][0];
            sorted_vectors[i][1] = this->neighbors.vectors[from][1];
            sorted_vectors[i][2] = this->neighbors.vectors[from][2];
        }
    }

    std::free(this->neighbors.pairs);
    this->neighbors.pairs = sorted_pairs;

    if (options.return_shifts) {
        std::free(this->neighbors.shifts);
        this->neighbors.shifts = sorted_shifts;
    }

    if (options.return_distances) {
        std::free(this->neighbors.distances);
        this->neighbors.distances = sorted_distances;
    }

    if (options.return_vectors) {
        std::free(this->neighbors.vectors);
        this->neighbors.vectors = sorted_vectors;
    }
}

void vesin::cpu::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCPU);

    if (neighbors.opaque != nullptr) {
        free_extra_data(neighbors);
    }

    std::free(neighbors.pairs);
    std::free(neighbors.shifts);
    std::free(neighbors.vectors);
    std::free(neighbors.distances);
}
