#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cpu_cell_list.hpp"
#include "verlet.hpp"

using namespace vesin;

void vesin::verlet_set_options(VerletState& state, VesinOptions options) {
    if (state.cutoff != options.cutoff || state.skin != options.skin || state.full_list != options.full) {
        state.has_cache = false;
        state.ref_positions.clear();
        state.pairs_i.clear();
        state.pairs_j.clear();
        state.shifts.clear();
        state.n_points = 0;
        state.n_pairs = 0;
        state.output_capacity = 0;
    }

    state.cutoff = options.cutoff;
    state.skin = options.skin;
    state.half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
    state.full_list = options.full;
}

bool vesin::verlet_needs_rebuild(
    const VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
) {
    if (!state.has_cache) {
        return true;
    }

    if (n_points != state.n_points) {
        return true;
    }

    for (int d = 0; d < 3; d++) {
        if (periodic[d] != state.ref_periodic[d]) {
            return true;
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(box[i][j] - state.ref_box[i][j]) > 1e-12) {
                return true;
            }
        }
    }

    for (size_t i = 0; i < n_points; i++) {
        double dx = points[i][0] - state.ref_positions[i * 3 + 0];
        double dy = points[i][1] - state.ref_positions[i * 3 + 1];
        double dz = points[i][2] - state.ref_positions[i * 3 + 2];
        double disp_sq = dx * dx + dy * dy + dz * dz;
        if (disp_sq > state.half_skin_sq) {
            return true;
        }
    }

    return false;
}

void vesin::verlet_rebuild(
    VerletState& state,
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3]
) {
    double expanded_cutoff = state.cutoff + state.skin;

    auto options = VesinOptions();
    options.cutoff = expanded_cutoff;
    options.full = state.full_list;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;
    options.skin = 0.0;

    VesinNeighborList tmp_neighbors;
    const char* rebuild_error = nullptr;
    int status = vesin_neighbors(
        points,
        n_points,
        box,
        periodic,
        VesinDevice{VesinCPU, 0},
        options,
        &tmp_neighbors,
        &rebuild_error
    );
    if (status != EXIT_SUCCESS) {
        std::string msg = "verlet_rebuild: ";
        if (rebuild_error) {
            msg += rebuild_error;
        }
        throw std::runtime_error(msg);
    }

    state.n_pairs = tmp_neighbors.length;
    state.pairs_i.resize(state.n_pairs);
    state.pairs_j.resize(state.n_pairs);
    state.shifts.resize(state.n_pairs * 3);

    for (size_t k = 0; k < state.n_pairs; k++) {
        state.pairs_i[k] = tmp_neighbors.pairs[k][0];
        state.pairs_j[k] = tmp_neighbors.pairs[k][1];
        state.shifts[k * 3 + 0] = tmp_neighbors.shifts[k][0];
        state.shifts[k * 3 + 1] = tmp_neighbors.shifts[k][1];
        state.shifts[k * 3 + 2] = tmp_neighbors.shifts[k][2];
    }

    state.n_points = n_points;
    state.ref_positions.resize(n_points * 3);
    std::memcpy(state.ref_positions.data(), points, n_points * 3 * sizeof(double));
    std::memcpy(state.ref_box, box, 9 * sizeof(double));
    for (int d = 0; d < 3; d++) {
        state.ref_periodic[d] = periodic[d];
    }

    state.has_cache = true;
    state.did_rebuild_flag = true;

    vesin_free(&tmp_neighbors);
}

void vesin::verlet_recompute(
    VerletState& state,
    const double (*points)[3],
    const double box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto matrix = Matrix{{{
        {{box[0][0], box[0][1], box[0][2]}},
        {{box[1][0], box[1][1], box[1][2]}},
        {{box[2][0], box[2][1], box[2][2]}},
    }}};
    auto bounding_box = BoundingBox(matrix, state.ref_periodic);

    double cutoff_sq = state.cutoff * state.cutoff;

    auto output_capacity = state.output_capacity;
    if (output_capacity == 0) {
        output_capacity = neighbors.length;
    }

    auto growable = cpu::GrowableNeighborList{neighbors, output_capacity, options};
    growable.reset();

    for (size_t k = 0; k < state.n_pairs; k++) {
        size_t i = state.pairs_i[k];
        size_t j = state.pairs_j[k];

        auto shift = CellShift{{
            state.shifts[k * 3 + 0],
            state.shifts[k * 3 + 1],
            state.shifts[k * 3 + 2],
        }};

        auto pi = Vector{{points[i][0], points[i][1], points[i][2]}};
        auto pj = Vector{{points[j][0], points[j][1], points[j][2]}};
        auto vec = pj - pi + shift.cartesian(bounding_box);
        double dist_sq = vec.dot(vec);

        if (dist_sq < cutoff_sq) {
            auto idx = growable.length();
            growable.set_pair(idx, i, j);

            if (options.return_shifts) {
                growable.set_shift(idx, shift);
            }

            if (options.return_distances) {
                growable.set_distance(idx, std::sqrt(dist_sq));
            }

            if (options.return_vectors) {
                growable.set_vector(idx, vec);
            }

            growable.increment_length();
        }
    }

    if (options.sorted) {
        growable.sort();
    }

    state.output_capacity = growable.capacity;
}
