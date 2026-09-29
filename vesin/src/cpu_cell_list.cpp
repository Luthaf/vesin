#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <limits>
#include <tuple>

#include "cpu_cell_list.hpp"
#include "threadpool.hpp"
#include "verlet.hpp"

using namespace vesin;
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

/// Measured cost of the cell list search, in microseconds. Searching around a
/// single point costs `cells_searched * (visit + points_per_cell * candidate)`.
struct SearchCostUs {
    /// Cost of visiting one neighbouring cell, whether that cell is empty or
    /// not.
    double visit = 0.0;
    /// Cost of checking one candidate point found inside a visited cell.
    double candidate = 0.0;
};

/// Time one serial pass of `foreach_neighbor_of` over all of `points`, and
/// return the elapsed time per `point * cells_searched` together with the
/// number of points per cell on that grid.
static std::pair<double, double> time_search(
    const std::vector<Vector>& points,
    double box_length,
    double cutoff
) {
    constexpr int32_t N_ITERS = 5;

    std::array<std::array<double, 3>, 3> matrix = {{
        {{box_length, 0.0, 0.0}},
        {{0.0, box_length, 0.0}},
        {{0.0, 0.0, box_length}},
    }};
    bool pbc[3] = {true, true, true};
    auto bounding_box = BoundingBox(matrix, pbc);

    CellListBuffersCPU buffers;
    auto cell_list = CellList(buffers, std::move(bounding_box), cutoff, points.size());
    cell_list.build(points.data(), points.size());

    auto cells_searched = static_cast<double>(
        (2 * cell_list.n_search()[0] + 1) *
        (2 * cell_list.n_search()[1] + 1) *
        (2 * cell_list.n_search()[2] + 1)
    );
    auto n_cells_total = cell_list.n_cells()[0] * cell_list.n_cells()[1] * cell_list.n_cells()[2];
    auto points_per_cell = static_cast<double>(points.size()) / static_cast<double>(n_cells_total);

    auto options = VesinOptions{};
    options.cutoff = cutoff;
    options.full = true;
    options.return_shifts = false;
    options.return_distances = false;
    options.return_vectors = false;

    double elapsed = 0.0;
    volatile size_t prevent_optimizations = 0;
    for (int32_t rep = 0; rep < N_ITERS; rep++) {
        auto start = std::chrono::high_resolution_clock::now();

        size_t count = 0;
        for (size_t si = 0; si < points.size(); si++) {
            cell_list.foreach_neighbor_of(si, options, [&](size_t, CellShift, Vector, double) { count++; });
        }
        prevent_optimizations += count;

        auto end = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration<double, std::micro>(end - start).count();
    }

    auto per_cell = (elapsed / N_ITERS) / (static_cast<double>(points.size()) * cells_searched);
    return std::make_pair(per_cell, points_per_cell);
}

static SearchCostUs measure_search_cost_us() {
    constexpr double CUTOFF = 3.0;
    constexpr size_t REPLICAS = 3;

    // a 3x3x3 supercell of diamond
    double A = 2.5222;
    auto unit_cell = std::vector<Vector>{
        {0.0,      0.0,      0.0},
        {0.25*A,   0.25*A,   0.25*A},
        {0.0,      0.5*A,    0.5*A},
        {0.25*A,   0.75*A,   0.75*A},
        {0.5*A,    0.0,      0.5*A},
        {0.75*A,   0.25*A,   0.75*A},
        {0.5*A,    0.5*A,    0.0},
        {0.75*A,   0.75*A,   0.25*A},
    };

    auto points = std::vector<Vector>();
    points.reserve(unit_cell.size() * REPLICAS * REPLICAS * REPLICAS);
    for (size_t x = 0; x < REPLICAS; x++) {
        for (size_t y = 0; y < REPLICAS; y++) {
            for (size_t z = 0; z < REPLICAS; z++) {
                for (const auto& point: unit_cell) {
                    points.push_back(Vector{
                        point[0] + A * static_cast<double>(x),
                        point[1] + A * static_cast<double>(y),
                        point[2] + A * static_cast<double>(z),
                    });
                }
            }
        }
    }
    auto box_length = A * static_cast<double>(REPLICAS);

    // Time the same points twice: once as is, where checking candidates
    // dominates, and once dilated so that almost every cell is empty and
    // visiting cells dominates.
    constexpr double DILATION = 4.0;
    auto dilated = points;
    for (auto& point: dilated) {
        point[0] *= DILATION;
        point[1] *= DILATION;
        point[2] *= DILATION;
    }

    auto [dense_cost, dense_points_per_cell] = time_search(points, box_length, CUTOFF);
    auto [sparse_cost, sparse_points_per_cell] = time_search(
        dilated, box_length * DILATION, CUTOFF
    );

    assert(dense_points_per_cell > sparse_points_per_cell);

    auto costs = SearchCostUs{};
    costs.candidate = (dense_cost - sparse_cost) / (dense_points_per_cell - sparse_points_per_cell);
    costs.visit = sparse_cost - sparse_points_per_cell * costs.candidate;

    // protect against very noisy measurements
    costs.candidate = std::max(costs.candidate, 0.0);
    costs.visit = std::max(costs.visit, 0.0);

    return costs;
}

void vesin::cpu::cell_list_neighbors(
    const Vector* points,
    size_t n_points,
    BoundingBox box,
    VesinOptions options,
    CellListBuffersCPU& buffers,
    VesinNeighborList& raw_neighbors,
    size_t& capacity
) {
    validate_algorithm(options);

    auto cell_list = CellList(buffers, std::move(box), options.cutoff, n_points);
    cell_list.build(points, n_points);

    auto& thread_pool = ThreadPool::global();

    // Determine the number of threads to use based on the number of points and
    // the estimated cost of parallelization. For small number of points (below
    // ~200), the overhead can be higher than just running serially.
    auto thread_count = std::max<size_t>(1, static_cast<size_t>(options.n_threads));
    thread_count = std::min(thread_count, n_points);

    auto n_cells_total = cell_list.n_cells()[0] * cell_list.n_cells()[1] * cell_list.n_cells()[2];
    if (n_cells_total > 0 && n_points > 0) {
        auto cells_searched = static_cast<double>(
            (2 * cell_list.n_search()[0] + 1) *
            (2 * cell_list.n_search()[1] + 1) *
            (2 * cell_list.n_search()[2] + 1)
        );
        auto mean_points_per_cell = static_cast<double>(n_points) / static_cast<double>(n_cells_total);

        static const auto SEARCH_COST_US = measure_search_cost_us();
        auto pass_cost = static_cast<double>(n_points)
                       * cells_searched
                       * (SEARCH_COST_US.visit + mean_points_per_cell * SEARCH_COST_US.candidate);

        // we do two passes: one to count the neighbors of each point, then one
        // more to write them out.
        constexpr size_t N_PASSES = 2;
        auto serial_cost = static_cast<double>(N_PASSES) * pass_cost;

        thread_count = pick_n_threads(serial_cost, N_PASSES, thread_count);
    }

    // Try to have at least `CHUNKS_PER_THREAD` chunks per thread up to
    // `MAX_CHUNK_SIZE` points per chunk.
    constexpr size_t CHUNKS_PER_THREAD = 4;
    constexpr size_t MAX_CHUNK_SIZE = 128;
    auto chunk_size = std::clamp<size_t>(
        n_points / (CHUNKS_PER_THREAD * std::max<size_t>(1, thread_count)), 1, MAX_CHUNK_SIZE
    );

    // First count all neighbours per point.
    buffers.count.assign(n_points + 1, 0);
    buffers.local_counts.resize(thread_count > 0 ? thread_count - 1 : 0);
    for (auto& local_count : buffers.local_counts) {
        local_count.assign(n_points + 1, 0);
    }

    thread_pool.run(thread_count, n_points, Scheduling::dynamic(chunk_size), [&](size_t sorted_i, size_t thread_id) {
        auto point_i = cell_list.original_index(sorted_i);
        size_t local_count = 0;
        cell_list.foreach_neighbor_of(sorted_i, options, [&](size_t, CellShift, Vector, double) {
            local_count++;
        });

        if (thread_id == 0) {
            buffers.count[point_i + 1] = local_count;
        } else {
            buffers.local_counts[thread_id - 1][point_i + 1] = local_count;
        }
    });

    for (auto& local_count : buffers.local_counts) {
        for (size_t i = 0; i <= n_points; i++) {
            buffers.count[i] += local_count[i];
        }
    }

    if (options.padded_to > 0) {
        assert(options.full);

        // Each point gets exactly `padded_to` slots in the output array, the
        // ones not used by actual neighbors are filled with padding in the
        // second pass.
        for (size_t i = 0; i < n_points; i++) {
            if (buffers.count[i + 1] > options.padded_to) {
                throw too_many_neighbors_error(i, options.padded_to);
            }
        }

        for (size_t i = 0; i <= n_points; i++) {
            buffers.count[i] = i * options.padded_to;
        }
    } else {
        // Get the write offset for each point in the output array by prefix sum.
        for (size_t i = 0; i < n_points; i++) {
            buffers.count[i + 1] += buffers.count[i];
        }
    }
    auto n_pairs = buffers.count[n_points];

    // Allocate output buffers at least as large as the retained capacity,
    // growing to `n_pairs` when needed.
    auto neighbors = GrowableNeighborList{raw_neighbors, capacity, options};
    neighbors.reset();
    neighbors.grow(n_pairs);

    // Actually fill the output buffers with the neighbour pairs. Each thread
    // writes to a separate offset in the output array, so no locks are needed.
    const auto& write_offset = buffers.count;

    thread_pool.run(thread_count, n_points, Scheduling::dynamic(chunk_size), [&](size_t sorted_i, size_t /*thread_id*/) {
        auto point_i = cell_list.original_index(sorted_i);
        auto offset = write_offset[point_i];

        cell_list.foreach_neighbor_of(sorted_i, options,
            [&](size_t original_j, CellShift shift, Vector vector, double distance2) {
                assert(offset < neighbors.capacity);
                raw_neighbors.pairs[offset][0] = point_i;
                raw_neighbors.pairs[offset][1] = original_j;
                if (options.return_shifts) {
                    raw_neighbors.shifts[offset][0] = shift[0];
                    raw_neighbors.shifts[offset][1] = shift[1];
                    raw_neighbors.shifts[offset][2] = shift[2];
                }
                if (options.return_distances) {
                    raw_neighbors.distances[offset] = std::sqrt(distance2);
                }
                if (options.return_vectors) {
                    raw_neighbors.vectors[offset][0] = vector[0];
                    raw_neighbors.vectors[offset][1] = vector[1];
                    raw_neighbors.vectors[offset][2] = vector[2];
                }
                offset++;
            }
        );

        // Without padding, the neighbors fill the whole range of this point
        // and this loop does nothing.
        auto end = write_offset[point_i + 1];
        assert(options.padded_to > 0 || offset == end);
        for (; offset < end; offset++) {
            set_padding_pair(raw_neighbors, offset, point_i, options);
        }
    });

    neighbors.neighbors.length = n_pairs;
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
        auto state = VerletList(extra.verlet_buffers);
        state.set_options(options);

        if (state.needs_rebuild(points, n_points, box)) {
            state.rebuild(points, n_points, box, extra.cell_list_buffers);
        }

        state.filter(points, box, options, raw_neighbors, extra.capacity);
    } else {
        cell_list_neighbors(
            points,
            n_points,
            std::move(box),
            options,
            extra.cell_list_buffers,
            raw_neighbors,
            extra.capacity
        );
    }
}

/* ========================================================================== */

/// Hard upper bound on the number of cells (2^21), to keep memory consumption
/// under control when the bounding box is much larger than the cutoff.
static constexpr double MAX_NUMBER_OF_CELLS = 1 << 21;

/// Also limit the number of cells to `2 x n_points`. This only matters when the
/// bounding box is mostly empty to prevent creating too many cells.
static constexpr double MAX_CELLS_PER_POINT = 2.0;

/// Shrink `n_cells` until the three dimensions multiply to `max_cells` or less.
static void fit_cells_in_budget(Vector& n_cells, double max_cells) {
    auto total = n_cells[0] * n_cells[1] * n_cells[2];
    if (total <= max_cells) {
        return;
    }

    // Scale all three at once to get close in one step, keeping roughly the
    // ratio of cells in each direction,
    auto ratio = std::cbrt(max_cells / total);
    for (size_t d = 0; d < 3; d++) {
        n_cells[d] = std::max(1.0, std::trunc(n_cells[d] * ratio));
    }

    // then walk the largest dimension down, since rounding each dimension
    // separately can leave the product just above the budget.
    total = n_cells[0] * n_cells[1] * n_cells[2];
    while (total > max_cells) {
        size_t largest = 0;
        for (size_t d = 1; d < 3; d++) {
            if (n_cells[d] > n_cells[largest]) {
                largest = d;
            }
        }

        if (n_cells[largest] <= 1.0) {
            break;
        }

        n_cells[largest] -= 1.0;
        total = n_cells[0] * n_cells[1] * n_cells[2];
    }
}

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

/// Build the canonical half of the cell-offset cube `[-n_search, n_search]^3`:
/// for every nonzero offset, keep exactly one of `{v, -v}` (the one whose
/// first nonzero component is positive). The excluded `(0, 0, 0)` offset is
/// handled separately, as the same-cell case.
static std::vector<std::array<int32_t, 3>> build_half_offsets(const std::array<int32_t, 3>& n_search) {
    std::vector<std::array<int32_t, 3>> offsets;
    offsets.reserve(
        (2 * n_search[0] + 1) * (2 * n_search[1] + 1) * (2 * n_search[2] + 1) / 2 + 1
    );
    for (int32_t dz = -n_search[2]; dz <= n_search[2]; dz++) {
        for (int32_t dy = -n_search[1]; dy <= n_search[1]; dy++) {
            for (int32_t dx = -n_search[0]; dx <= n_search[0]; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                bool positive = (dx > 0) || (dx == 0 && dy > 0) || (dx == 0 && dy == 0 && dz > 0);
                if (positive) {
                    offsets.push_back({dx, dy, dz});
                }
            }
        }
    }
    return offsets;
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

CellList::CellList(CellListBuffersCPU& buffers, BoundingBox box, double cutoff, size_t n_points):
    buffers_(buffers),
    n_search_({0, 0, 0}),
    n_cells_({0, 0, 0}),
    box_(std::move(box)) {
    auto distances_between_faces = box_.distances_between_faces();

    // limit memory consumption, and the `O(n_cells)` part of `build()`
    auto max_cells = std::min(
        MAX_NUMBER_OF_CELLS,
        std::max(1.0, static_cast<double>(n_points) * MAX_CELLS_PER_POINT)
    );

    size_t n_periodic = 0;
    for (size_t d = 0; d < 3; d++) {
        if (box_.periodic(d)) {
            n_periodic += 1;
        }
    }

    // Pick how many cells to put across one cutoff.
    //
    // Cells one cutoff wide is the textbook choice, but then the search sweeps
    // a 3x3x3 block of them, i.e. (3 * cutoff)^3 of volume, where the cutoff
    // sphere only occupies (4 pi / 3) * cutoff^3 -- a factor of 6.4 more
    // candidate pairs than there are neighbors to find. Splitting each cutoff
    // into `k` cells brings the swept span per dimension down from 3 * cutoff
    // towards the (2 + 1/k) * cutoff floor, at the price of (2k+1)^3 cell
    // visits instead of 27.
    //
    // That trade only pays off when the cells hold enough atoms for the
    // candidates to dominate the cell visits. Minimising
    //
    //     cost(k) = atoms_per_cell * (2 + 1/k)^3 + cell_visit_cost * (2k+1)^3
    //
    // over k gives k = (atoms_per_cell / (2 * cell_visit_cost))^(1/4), where
    // `atoms_per_cell` is measured on the one-cell-per-cutoff grid.


    // Cost estimate of visiting one extra cell, in units of the cost of
    // checking one extra candidate pair.
    constexpr double CELL_VISIT_COST = 8.0;
    // Limit the number of cells we use when the box is a lot smaller than the
    // cutoff
    constexpr int32_t MAX_CELLS_PER_CUTOFF = 6;

    auto points_per_cell = static_cast<double>(n_points);
    for (size_t d = 0; d < 3; d++) {
        points_per_cell /= std::clamp(
            std::trunc(distances_between_faces[d] / cutoff), 1.0, max_cells
        );
    }

    auto k = std::pow(points_per_cell / (2.0 * CELL_VISIT_COST), 0.25);
    auto cells_per_cutoff = std::clamp(
        static_cast<int32_t>(std::lround(k)), 1, MAX_CELLS_PER_CUTOFF
    );

    auto n_cells = Vector{0.0, 0.0, 0.0};
    for (size_t d = 0; d < 3; d++) {
        // Refining a non-periodic direction under mixed periodicity buys
        // nothing: `n_search` there is widened below to cover the whole extent
        // regardless, so a finer grid sweeps the same points in more cells.
        auto dim_cells_per_cutoff = cells_per_cutoff;
        if (n_periodic > 0 && n_periodic < 3 && !box_.periodic(d)) {
            dim_cells_per_cutoff = 1;
        }

        n_cells[d] = std::clamp(
            std::trunc(dim_cells_per_cutoff * distances_between_faces[d] / cutoff),
            1.0,
            max_cells
        );
    }

    assert(std::isfinite(n_cells[0]) && std::isfinite(n_cells[1]) && std::isfinite(n_cells[2]));

    fit_cells_in_budget(n_cells, max_cells);

    // number of cells to search in each direction to make sure all possible
    // pairs below the cutoff are accounted for.
    this->n_search_ = std::array<int32_t, 3>{
        static_cast<int32_t>(std::ceil(cutoff * n_cells[0] / distances_between_faces[0])),
        static_cast<int32_t>(std::ceil(cutoff * n_cells[1] / distances_between_faces[1])),
        static_cast<int32_t>(std::ceil(cutoff * n_cells[2] / distances_between_faces[2])),
    };

    this->n_cells_ = std::array<size_t, 3>{
        static_cast<size_t>(n_cells[0]),
        static_cast<size_t>(n_cells[1]),
        static_cast<size_t>(n_cells[2]),
    };

    for (size_t spatial = 0; spatial < 3; spatial++) {
        if (n_search_[spatial] < 1) {
            n_search_[spatial] = 1;
        }
    }

    for (size_t spatial = 0; spatial < 3; spatial++) {
        if (box_.periodic(spatial)) {
            continue;
        }

        // A non-periodic direction never searches past its own extent, so we
        // can reduce the search size.
        auto extent = static_cast<int32_t>(n_cells_[spatial]) - 1;
        n_search_[spatial] = n_periodic > 0 ? extent : std::min(n_search_[spatial], extent);
    }

    // Cells are only all the same parallelepiped when every direction wraps the
    // same way; under mixed periodicity the non-periodic directions are
    // searched to their full extent instead.
    can_prune_offsets_ = (n_periodic == 3 || n_periodic == 0);
    if (n_periodic == 3) {
        const auto& matrix = box_.matrix();
        for (size_t d = 0; d < 3; d++) {
            cell_vectors_[d] = Vector{matrix[d]} * (1.0 / n_cells[d]);
        }
    } else if (n_periodic == 0) {
        for (size_t d = 0; d < 3; d++) {
            cell_vectors_[d] = Vector{0.0, 0.0, 0.0};
            cell_vectors_[d][d] = distances_between_faces[d] / n_cells[d];
        }
    }

    half_offsets_ = build_half_offsets(n_search_);

    if (can_prune_offsets_) {
        auto cutoff2 = cutoff * cutoff;
        auto end = std::remove_if(
            half_offsets_.begin(),
            half_offsets_.end(),
            [&](const std::array<int32_t, 3>& offset) {
                return this->offset_out_of_reach(offset[0], offset[1], offset[2], cutoff2);
            }
        );
        half_offsets_.erase(end, half_offsets_.end());

        auto n_offsets = static_cast<size_t>(2 * n_search_[0] + 1)
                       * static_cast<size_t>(2 * n_search_[1] + 1)
                       * static_cast<size_t>(2 * n_search_[2] + 1);
        cell_out_of_reach_.assign(n_offsets, 0);

        for (int32_t dz = -n_search_[2]; dz <= n_search_[2]; dz++) {
            for (int32_t dy = -n_search_[1]; dy <= n_search_[1]; dy++) {
                for (int32_t dx = -n_search_[0]; dx <= n_search_[0]; dx++) {
                    if (this->offset_out_of_reach(dx, dy, dz, cutoff2)) {
                        cell_out_of_reach_[this->offset_index(dx, dy, dz)] = 1;
                    }
                }
            }
        }
    }
}

std::pair<size_t, vesin::CellShift> CellList::point_cell(Vector position) const {
    auto fractional = box_.cartesian_to_fractional(position);

    // find the cell in which this point should go
    auto cell_index = std::array<int32_t, 3>{
        static_cast<int32_t>(std::floor(fractional[0] * static_cast<double>(n_cells_[0]))),
        static_cast<int32_t>(std::floor(fractional[1] * static_cast<double>(n_cells_[1]))),
        static_cast<int32_t>(std::floor(fractional[2] * static_cast<double>(n_cells_[2]))),
    };

    // deal with pbc by wrapping the point inside if it was outside of the cell
    auto [shift, remainder] = divmod(cell_index, n_cells_);
    for (size_t spatial = 0; spatial < 3; spatial++) {
        assert(box_.periodic(spatial) || shift[spatial] == 0);
    }

    auto linear = (n_cells_[0] * n_cells_[1] * remainder[2])
                + (n_cells_[0] * remainder[1])
                + remainder[0];
    return {linear, CellShift(shift)};
}

void CellList::build(const Vector* points, size_t n_points) {
    auto total_cells = n_cells_[0] * n_cells_[1] * n_cells_[2];

    // count the number of points per cell.
    buffers_.cell_count.assign(total_cells, 0);
    for (size_t i = 0; i < n_points; i++) {
        buffers_.cell_count[point_cell(points[i]).first]++;
    }

    // Prefix sum -> per-cell start offsets into the sorted arrays.
    buffers_.cell_start.resize(total_cells);
    size_t offset = 0;
    for (size_t c = 0; c < total_cells; c++) {
        buffers_.cell_start[c] = offset;
        offset += buffers_.cell_count[c];
    }

    // scatter each point into its final CSR slot.
    buffers_.original_index.resize(offset);
    buffers_.sorted_shifts.resize(offset);
    auto next = buffers_.cell_start;
    for (size_t i = 0; i < n_points; i++) {
        auto [cell, shift] = point_cell(points[i]);
        auto s = next[cell]++;
        buffers_.original_index[s] = i;
        buffers_.sorted_shifts[s] = shift;
    }

    // Precompute wrapped positions: wrapped_pos[s] = pos - wrap_shift
    buffers_.wrapped_positions.resize(n_points);
    for (size_t s = 0; s < n_points; s++) {
        buffers_.wrapped_positions[s] = points[buffers_.original_index[s]] - buffers_.sorted_shifts[s].cartesian(box_);
    }
}

template <typename Function>
void CellList::foreach_neighbor_of_half(size_t sorted_i, VesinOptions options, Function callback) const {
    auto cutoff2 = options.cutoff * options.cutoff;

    // Find cell coordinates by looking up which cell contains si.
    // Since points are sorted by cell, binary search cell_start_ for si.
    auto it = std::upper_bound(buffers_.cell_start.begin(), buffers_.cell_start.end(), sorted_i);
    auto cell_idx = static_cast<size_t>(it - buffers_.cell_start.begin()) - 1;

    auto cell_idx_x = static_cast<int32_t>(cell_idx % n_cells_[0]);
    auto cell_idx_y = static_cast<int32_t>((cell_idx / n_cells_[0]) % n_cells_[1]);
    auto cell_idx_z = static_cast<int32_t>(cell_idx / (n_cells_[0] * n_cells_[1]));

    auto shift_i = buffers_.sorted_shifts[sorted_i];
    const auto& wrapped_i = buffers_.wrapped_positions[sorted_i];
    const auto& matrix = box_.matrix();

    // Same-cell pairs
    auto end_same = buffers_.cell_start[cell_idx] + buffers_.cell_count[cell_idx];
    for (size_t sorted_j = sorted_i + 1; sorted_j < end_same; sorted_j++) {
        auto vector = buffers_.wrapped_positions[sorted_j] - wrapped_i;
        auto distance2 = vector.dot(vector);
        if (distance2 >= cutoff2) {
            continue;
        }

        auto point_j = buffers_.original_index[sorted_j];
        auto shift = shift_i - buffers_.sorted_shifts[sorted_j];
        callback(point_j, shift, vector, distance2);
    }

    // Cross-cell pairs: `half_offsets_` keeps exactly one of `{v, -v}` for
    // every nonzero cell offset, so every cell-to-cell pair of points is
    // visited exactly once overall (from the cell reached by the canonical
    // direction).
    for (const auto& offset : half_offsets_) {
        int32_t cell_shift_x = 0;
        int32_t nx = cell_idx_x + offset[0];
        if (box_.periodic(0)) {
            auto [q, r] = divmod(nx, static_cast<int32_t>(n_cells_[0]));
            cell_shift_x = q;
            nx = r;
        } else if (nx < 0 || nx >= static_cast<int32_t>(n_cells_[0])) {
            continue;
        }

        int32_t cell_shift_y = 0;
        int32_t ny = cell_idx_y + offset[1];
        if (box_.periodic(1)) {
            auto [q, r] = divmod(ny, static_cast<int32_t>(n_cells_[1]));
            cell_shift_y = q;
            ny = r;
        } else if (ny < 0 || ny >= static_cast<int32_t>(n_cells_[1])) {
            continue;
        }

        int32_t cell_shift_z = 0;
        int32_t nz = cell_idx_z + offset[2];
        if (box_.periodic(2)) {
            auto [q, r] = divmod(nz, static_cast<int32_t>(n_cells_[2]));
            cell_shift_z = q;
            nz = r;
        } else if (nz < 0 || nz >= static_cast<int32_t>(n_cells_[2])) {
            continue;
        }

        auto cell_shift_vec = Vector{matrix[0]} * static_cast<double>(cell_shift_x)
                             + Vector{matrix[1]} * static_cast<double>(cell_shift_y)
                             + Vector{matrix[2]} * static_cast<double>(cell_shift_z);

        size_t neighbor_cell_idx = (n_cells_[0] * n_cells_[1] * static_cast<size_t>(nz))
                                 + (n_cells_[0] * static_cast<size_t>(ny))
                                 + static_cast<size_t>(nx);

        size_t begin_j = buffers_.cell_start[neighbor_cell_idx];
        size_t end_j = begin_j + buffers_.cell_count[neighbor_cell_idx];

        for (size_t sorted_j = begin_j; sorted_j < end_j; sorted_j++) {
            auto vector = buffers_.wrapped_positions[sorted_j] - wrapped_i + cell_shift_vec;
            auto distance2 = vector.dot(vector);

            if (distance2 >= cutoff2) {
                continue;
            }

            auto point_j = buffers_.original_index[sorted_j];
            auto shift = CellShift(cell_shift_x, cell_shift_y, cell_shift_z) + shift_i - buffers_.sorted_shifts[sorted_j];

            callback(point_j, shift, vector, distance2);
        }
    }
}

template <typename Function>
void CellList::foreach_neighbor_of_full(size_t sorted_i, VesinOptions options, Function callback) const {
    auto cutoff2 = options.cutoff * options.cutoff;

    // Find cell coordinates by looking up which cell contains si.
    // Since points are sorted by cell, binary search cell_start_ for si.
    auto it = std::upper_bound(buffers_.cell_start.begin(), buffers_.cell_start.end(), sorted_i);
    auto cell_idx = static_cast<size_t>(it - buffers_.cell_start.begin()) - 1;

    auto cell_idx_x = static_cast<int32_t>(cell_idx % n_cells_[0]);
    auto cell_idx_y = static_cast<int32_t>((cell_idx / n_cells_[0]) % n_cells_[1]);
    auto cell_idx_z = static_cast<int32_t>(cell_idx / (n_cells_[0] * n_cells_[1]));

    auto shift_i = buffers_.sorted_shifts[sorted_i];
    const auto& wrapped_i = buffers_.wrapped_positions[sorted_i];
    const auto& matrix = box_.matrix();

    for (int32_t dz = -n_search_[2]; dz <= n_search_[2]; dz++) {
        int32_t cell_shift_z = 0;
        int32_t nz = cell_idx_z + dz;
        if (box_.periodic(2)) {
            auto [q, r] = divmod(nz, static_cast<int32_t>(n_cells_[2]));
            cell_shift_z = q;
            nz = r;
        } else if (nz < 0 || nz >= static_cast<int32_t>(n_cells_[2])) {
            continue;
        }
        auto partial_shift_z = Vector{matrix[2]} * static_cast<double>(cell_shift_z);

        for (int32_t dy = -n_search_[1]; dy <= n_search_[1]; dy++) {
            int32_t cell_shift_y = 0;
            int32_t ny = cell_idx_y + dy;
            if (box_.periodic(1)) {
                auto [q, r] = divmod(ny, static_cast<int32_t>(n_cells_[1]));
                cell_shift_y = q;
                ny = r;
            } else if (ny < 0 || ny >= static_cast<int32_t>(n_cells_[1])) {
                continue;
            }
            auto partial_shift_yz = partial_shift_z + Vector{matrix[1]} * static_cast<double>(cell_shift_y);

            for (int32_t dx = -n_search_[0]; dx <= n_search_[0]; dx++) {
                if (can_prune_offsets_ && cell_out_of_reach_[this->offset_index(dx, dy, dz)]) {
                    continue;
                }

                int32_t cell_shift_x = 0;
                int32_t nx = cell_idx_x + dx;
                if (box_.periodic(0)) {
                    auto [q, r] = divmod(nx, static_cast<int32_t>(n_cells_[0]));
                    cell_shift_x = q;
                    nx = r;
                } else if (nx < 0 || nx >= static_cast<int32_t>(n_cells_[0])) {
                    continue;
                }
                auto cell_shift_vec = partial_shift_yz + Vector{matrix[0]} * static_cast<double>(cell_shift_x);

                size_t neighbor_cell_idx = (n_cells_[0] * n_cells_[1] * nz)
                                         + (n_cells_[0] * ny)
                                         + nx;

                size_t begin_j = buffers_.cell_start[neighbor_cell_idx];
                size_t end_j = begin_j + buffers_.cell_count[neighbor_cell_idx];

                for (size_t sorted_j = begin_j; sorted_j < end_j; sorted_j++) {
                    // Skip self in the same cell early (before distance).
                    if (sorted_j == sorted_i && dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }

                    auto vector = buffers_.wrapped_positions[sorted_j] - wrapped_i + cell_shift_vec;
                    auto distance2 = vector.dot(vector);

                    if (distance2 >= cutoff2) {
                        continue;
                    }

                    // Compute shift only for pairs that pass the cutoff.
                    auto point_j = buffers_.original_index[sorted_j];
                    auto shift = CellShift(cell_shift_x, cell_shift_y, cell_shift_z)
                               + shift_i - buffers_.sorted_shifts[sorted_j];

                    callback(point_j, shift, vector, distance2);
                }
            }
        }
    }
}

/* ========================================================================== */

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

    if (new_size <= this->capacity) {
        return;
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
