#include <algorithm>
#include <cmath>
#include <cstring>

#include "cpu_cell_list.hpp"
#include "verlet.hpp"

using namespace vesin;

// Sum of the two largest box-corner displacements, used to shrink the Verlet
// rebuild threshold when the box deforms (e.g. NPT). 
static double corner_point_displacements(const Matrix& box, const Matrix& ref_box) {
    double delta1 = 0.0;
    double delta2 = 0.0;
    for (size_t i = 0; i < 8; i++) {
        auto bits = Vector{
            static_cast<double>(i & 1),
            static_cast<double>((i >> 1) & 1),
            static_cast<double>((i >> 2) & 1),
        };
        auto displacement = bits * box - bits * ref_box;
        double delta = displacement.dot(displacement);
        if (delta > delta1) {
            delta2 = delta1;
            delta1 = delta;
        } else if (delta > delta2) {
            delta2 = delta;
        }
    }
    return std::sqrt(delta1) + std::sqrt(delta2);
}

cpu::VerletList::~VerletList() {
    this->clear_candidates();
}

void cpu::VerletList::clear_candidates() {
    if (candidates_.device.type == VesinCPU) {
        cpu::free_neighbors(candidates_);
    }

    candidates_ = VesinNeighborList();
    has_cache_ = false;
    ref_points_.clear();
}

void cpu::VerletList::set_options(VesinOptions options) {
    if (options_.cutoff != options.cutoff || options_.skin != options.skin || options_.full != options.full) {
        this->clear_candidates();
    }

    options_ = options;
    half_skin_sq_ = (options.skin / 2.0) * (options.skin / 2.0);
}

bool cpu::VerletList::needs_rebuild(
    const Vector* points,
    size_t n_points,
    const BoundingBox& box
) const {
    if (!has_cache_) {
        return true;
    }

    if (n_points != ref_points_.size()) {
        return true;
    }

    for (size_t d = 0; d < 3; d++) {
        if (box.periodic(d) != ref_periodic_[d]) {
            return true;
        }
    }
    // The corner displacement uses the full box matrix: non-periodic directions
    // carry no shift, so including them only over-estimates the bound (more
    // rebuilds, never fewer). This stays correct for any mix of periodicity.
    auto half_displacement = corner_point_displacements(box.matrix(), ref_matrix_) / 2.0;
    if (half_displacement * half_displacement > half_skin_sq_) {
        return true;
    }

    // Verlet-list reuse is valid while every atom stays within skin/2 of its
    // reference position: any pair inside cutoff is present in the cached
    // candidate list built at cutoff + skin. See Verlet, Phys. Rev. 159, 98-103
    // (1967), doi:10.1103/PhysRev.159.98, and Chialvo and Debenedetti, Comput.
    // Phys. Commun. 60, 215-224 (1990), doi:10.1016/0010-4655(90)90007-N.
    // One can also take the change of the box into account, see the 
    // implementation of LAMMPS:
    // https://github.com/lammps/lammps/blob/3bfc12b02799eedf79d779d66fad8c4c60554084/src/neighbor.cpp#L2434-L2448
    auto half_threshold_sq = (sqrt(half_skin_sq_) - half_displacement) * (sqrt(half_skin_sq_) - half_displacement);
    for (size_t i = 0; i < n_points; i++) {
        auto displacement = points[i] - ref_points_[i];
        double displacement_sq = displacement.dot(displacement);
        if (displacement_sq > half_threshold_sq) {
            return true;
        }
    }

    return false;
}

void cpu::VerletList::rebuild(
    const Vector* points,
    size_t n_points,
    const BoundingBox& box
) {
    this->clear_candidates();

    auto build_options = VesinOptions();
    build_options.cutoff = options_.cutoff + options_.skin;
    build_options.full = options_.full;
    build_options.sorted = false;
    build_options.algorithm = VesinCellList;
    build_options.return_shifts = true;
    build_options.return_distances = false;
    build_options.return_vectors = false;
    build_options.skin = 0.0;

    candidates_.device = {VesinCPU, 0};

    auto periodic = std::array<bool, 3>{box.periodic(0), box.periodic(1), box.periodic(2)};
    auto candidate_box = BoundingBox(box.matrix(), periodic.data());
    candidate_box.make_bounding_for(reinterpret_cast<const double (*)[3]>(points), n_points);

    size_t candidate_capacity = 0;
    cpu::cell_list_neighbors(points, n_points, std::move(candidate_box), build_options, candidates_, candidate_capacity);

    ref_points_.resize(n_points);
    std::memcpy(ref_points_.data(), points, n_points * sizeof(Vector));
    ref_matrix_ = box.matrix();
    for (size_t d = 0; d < 3; d++) {
        ref_periodic_[d] = box.periodic(d);
    }

    has_cache_ = true;
}

void cpu::VerletList::filter(
    const Vector* points,
    const BoundingBox& box,
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t& output_capacity
) const {
    double cutoff_sq = options_.cutoff * options_.cutoff;

    auto initial_capacity = std::max(output_capacity, neighbors.length);

    auto growable = cpu::GrowableNeighborList{neighbors, initial_capacity, options};
    growable.reset();

    // The cached list is an over-complete Verlet candidate list. Each call
    // filters candidates with the exact cutoff and requested shift/vector outputs.
    for (size_t k = 0; k < candidates_.length; k++) {
        size_t i = candidates_.pairs[k][0];
        size_t j = candidates_.pairs[k][1];

        auto shift = CellShift(candidates_.shifts[k]);

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

    output_capacity = growable.capacity;
}
