#include <algorithm>
#include <cmath>
#include <cstring>

#include "cpu_cell_list.hpp"
#include "verlet.hpp"

using namespace vesin;

static BoundingBox make_box_like(const BoundingBox& box, const Vector* points, size_t n_points) {
    auto periodic = std::array<bool, 3>{box.periodic(0), box.periodic(1), box.periodic(2)};
    auto candidate_box = BoundingBox(box.matrix(), periodic.data());
    candidate_box.make_bounding_for(reinterpret_cast<const double (*)[3]>(points), n_points);
    return candidate_box;
}

namespace {

Vector fractional_displacement_to_cartesian(Vector displacement, const BoundingBox& box) {
    auto cartesian = displacement * box.matrix();
    auto distances = box.distances_between_faces();

    for (size_t spatial = 0; spatial < 3; spatial++) {
        if (!box.periodic(spatial)) {
            cartesian[spatial] = displacement[spatial] * distances[spatial];
        }
    }

    return cartesian;
}

Vector minimum_image_displacement(const Vector& point, const Vector& reference, const BoundingBox& box) {
    auto point_fractional = box.cartesian_to_fractional(point);
    auto reference_fractional = box.cartesian_to_fractional(reference);

    auto delta = point_fractional - reference_fractional;
    for (size_t spatial = 0; spatial < 3; spatial++) {
        if (!box.periodic(spatial)) {
            continue;
        }

        while (delta[spatial] > 0.5) {
            delta[spatial] -= 1.0;
        }
        while (delta[spatial] < -0.5) {
            delta[spatial] += 1.0;
        }
    }

    return fractional_displacement_to_cartesian(delta, box);
}

} // namespace

cpu::VerletState::~VerletState() {
    this->clear_candidates();
}

void cpu::VerletState::clear_candidates() {
    if (this->candidates.device.type == VesinCPU) {
        cpu::free_neighbors(this->candidates);
    }

    this->candidates = VesinNeighborList();
    this->has_cache = false;
    this->ref_positions.clear();
    this->n_points = 0;
}

void cpu::VerletState::set_options(VesinOptions options) {
    if (this->cutoff != options.cutoff || this->skin != options.skin || this->full_list != options.full) {
        this->clear_candidates();
        this->output_capacity = 0;
    }

    this->cutoff = options.cutoff;
    this->skin = options.skin;
    this->half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
    this->full_list = options.full;
}

bool cpu::VerletState::needs_rebuild(
    const Vector* points,
    size_t n_points,
    const BoundingBox& box
) const {
    if (!this->has_cache) {
        return true;
    }

    if (n_points != this->n_points) {
        return true;
    }

    for (size_t d = 0; d < 3; d++) {
        if (box.periodic(d) != this->ref_periodic[d]) {
            return true;
        }
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (std::abs(box.matrix()[i][j] - this->ref_matrix[i][j]) > 1e-12) {
                return true;
            }
        }
    }

    // Verlet-list reuse is valid while every atom stays within skin/2 of its
    // reference position: any pair inside cutoff is present in the cached
    // candidate list built at cutoff + skin. See Verlet, Phys. Rev. 159, 98-103
    // (1967), doi:10.1103/PhysRev.159.98, and Chialvo and Debenedetti, Comput.
    // Phys. Commun. 60, 215-224 (1990), doi:10.1016/0010-4655(90)90007-N.
    for (size_t i = 0; i < n_points; i++) {
        auto delta = minimum_image_displacement(
            points[i],
            Vector{
                this->ref_positions[i * 3 + 0],
                this->ref_positions[i * 3 + 1],
                this->ref_positions[i * 3 + 2],
            },
            box
        );
        auto disp_sq = delta.dot(delta);
        if (disp_sq > this->half_skin_sq) {
            return true;
        }
    }

    return false;
}

void cpu::VerletState::rebuild(
    const Vector* points,
    size_t n_points,
    const BoundingBox& box
) {
    this->clear_candidates();

    auto options = VesinOptions();
    options.cutoff = this->cutoff + this->skin;
    options.full = this->full_list;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;
    options.skin = 0.0;

    this->candidates.device = {VesinCPU, 0};
    auto candidate_box = make_box_like(box, points, n_points);
    cpu::stateless_neighbors(points, n_points, std::move(candidate_box), options, this->candidates);

    this->n_points = n_points;
    this->ref_positions.resize(n_points * 3);
    std::memcpy(this->ref_positions.data(), points, n_points * 3 * sizeof(double));
    this->ref_matrix = box.matrix();
    for (size_t d = 0; d < 3; d++) {
        this->ref_periodic[d] = box.periodic(d);
    }

    this->has_cache = true;
    this->did_rebuild_flag = true;
}

void cpu::VerletState::recompute(
    const Vector* points,
    const BoundingBox& box,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    double cutoff_sq = this->cutoff * this->cutoff;

    // Reuse the largest observed caller-visible output capacity so temporary
    // output dips do not trigger repeated reallocation during long simulations.
    auto output_capacity = std::max(this->output_capacity, neighbors.length);

    auto growable = cpu::GrowableNeighborList{neighbors, output_capacity, options};
    growable.reset();

    // The cached list is an over-complete Verlet candidate list. Each call
    // filters candidates with the exact cutoff and requested shift/vector outputs.
    for (size_t k = 0; k < this->candidates.length; k++) {
        size_t i = this->candidates.pairs[k][0];
        size_t j = this->candidates.pairs[k][1];

        auto shift = CellShift{{
            this->candidates.shifts[k][0],
            this->candidates.shifts[k][1],
            this->candidates.shifts[k][2],
        }};

        auto vec = points[j] - points[i] + shift.cartesian(box);
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

    // Keep the maximum observed output capacity for next recompute.
    this->output_capacity = std::max(this->output_capacity, growable.capacity);
}
