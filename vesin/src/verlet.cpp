#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "hwy/highway.h"

#include "cluster.hpp"
#include "cpu_cell_list.hpp"
#include "verlet.hpp"

using namespace vesin;

namespace {
namespace hn = hwy::HWY_NAMESPACE;

static BoundingBox make_box_like(const BoundingBox& box, const Vector* points, size_t n_points) {
    auto periodic = std::array<bool, 3>{box.periodic(0), box.periodic(1), box.periodic(2)};
    auto candidate_box = BoundingBox(box.matrix(), periodic.data());
    candidate_box.make_bounding_for(reinterpret_cast<const double (*)[3]>(points), n_points);
    return candidate_box;
}

static bool is_zero_shift(CellShift shift) {
    return shift[0] == 0 && shift[1] == 0 && shift[2] == 0;
}

static Vector shift_cartesian(CellShift shift, const BoundingBox& box) {
    if (is_zero_shift(shift)) {
        return Vector{0.0, 0.0, 0.0};
    }
    return shift.cartesian(box);
}

static std::vector<cpu::VerletCandidateBlock> pack_simd_candidate_blocks(
    const VesinNeighborList& candidates,
    const std::vector<Vector>& shift_vectors,
    size_t& candidate_length
) {
    auto blocks = std::vector<cpu::VerletCandidateBlock>();
    candidate_length = candidates.length;
    blocks.reserve((candidates.length + CLUSTER_SIZE_CPU - 1) / CLUSTER_SIZE_CPU);

    auto block = cpu::VerletCandidateBlock();
    for (size_t k = 0; k < candidates.length; k++) {
        auto lane = block.count;
        block.first[lane] = candidates.pairs[k][0];
        block.second[lane] = candidates.pairs[k][1];
        block.shifts[lane] = CellShift{{
            candidates.shifts[k][0],
            candidates.shifts[k][1],
            candidates.shifts[k][2],
        }};
        block.shift_x[lane] = shift_vectors[k][0];
        block.shift_y[lane] = shift_vectors[k][1];
        block.shift_z[lane] = shift_vectors[k][2];
        block.count += 1;

        if (block.count == CLUSTER_SIZE_CPU) {
            blocks.push_back(block);
            block = cpu::VerletCandidateBlock();
        }
    }

    if (block.count != 0) {
        blocks.push_back(block);
    }

    return blocks;
}

HWY_ATTR
static void simd_filter_deltas(
    const double* HWY_RESTRICT dx,
    const double* HWY_RESTRICT dy,
    const double* HWY_RESTRICT dz,
    double cutoff_sq,
    double* HWY_RESTRICT dist_sq,
    uint8_t* HWY_RESTRICT mask
) {
    const hn::ScalableTag<double> d;
    const size_t N = hn::Lanes(d);
    const auto vcut = hn::Set(d, cutoff_sq);

    for (size_t lane = 0; lane < CLUSTER_SIZE_CPU; lane += N) {
        auto vdx = hn::Load(d, dx + lane);
        auto vdy = hn::Load(d, dy + lane);
        auto vdz = hn::Load(d, dz + lane);
        auto vdist = hn::MulAdd(vdx, vdx, hn::MulAdd(vdy, vdy, hn::Mul(vdz, vdz)));
        auto vmask = hn::Lt(vdist, vcut);

        hn::Store(vdist, d, dist_sq + lane);

        uint8_t bits_buf[8] = {};
        hn::StoreMaskBits(d, vmask, bits_buf);
        uint8_t bits = bits_buf[0];
        for (size_t k = 0; k < N && (lane + k) < CLUSTER_SIZE_CPU; k++) {
            mask[lane + k] = (bits >> k) & 1;
        }
    }
}

static void filter_simd_candidate_blocks(
    const Vector* points,
    const std::vector<cpu::VerletCandidateBlock>& blocks,
    double cutoff_sq,
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t initial_capacity,
    size_t& output_capacity
) {
    auto growable = cpu::GrowableNeighborList{neighbors, initial_capacity, options};
    growable.reset();

    alignas(64) double dx[CLUSTER_SIZE_CPU];
    alignas(64) double dy[CLUSTER_SIZE_CPU];
    alignas(64) double dz[CLUSTER_SIZE_CPU];
    alignas(64) double dist_sq[CLUSTER_SIZE_CPU];
    uint8_t mask[CLUSTER_SIZE_CPU];

    for (const auto& block : blocks) {
        for (size_t lane = 0; lane < block.count; lane++) {
            auto i = block.first[lane];
            auto j = block.second[lane];
            dx[lane] = points[j][0] - points[i][0] + block.shift_x[lane];
            dy[lane] = points[j][1] - points[i][1] + block.shift_y[lane];
            dz[lane] = points[j][2] - points[i][2] + block.shift_z[lane];
        }
        for (size_t lane = block.count; lane < CLUSTER_SIZE_CPU; lane++) {
            dx[lane] = std::numeric_limits<double>::infinity();
            dy[lane] = 0.0;
            dz[lane] = 0.0;
        }

        simd_filter_deltas(dx, dy, dz, cutoff_sq, dist_sq, mask);

        for (size_t lane = 0; lane < block.count; lane++) {
            if (!mask[lane]) {
                continue;
            }

            auto index = growable.length();
            growable.set_pair(index, block.first[lane], block.second[lane]);

            if (options.return_shifts) {
                growable.set_shift(index, block.shifts[lane]);
            }

            if (options.return_distances) {
                growable.set_distance(index, std::sqrt(dist_sq[lane]));
            }

            if (options.return_vectors) {
                growable.set_vector(index, Vector{dx[lane], dy[lane], dz[lane]});
            }

            growable.increment_length();
        }
    }

    if (options.sorted) {
        growable.sort();
    }

    output_capacity = growable.capacity;
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
    this->candidate_shift_vectors.clear();
    this->simd_candidate_blocks.clear();
    this->simd_candidate_length = 0;
    this->cluster_grid = ClusterGrid();
    this->cluster_candidates.clear();
    this->use_cluster_candidates = false;
    this->has_cache = false;
    this->ref_positions.clear();
    this->n_points = 0;
}

void cpu::VerletState::set_options(VesinOptions options) {
    if (this->options.cutoff != options.cutoff || this->options.skin != options.skin || this->options.full != options.full ||
        this->options.algorithm != options.algorithm) {
        this->clear_candidates();
    }

    this->options = options;
    this->half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
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
        double dx = points[i][0] - this->ref_positions[i * 3 + 0];
        double dy = points[i][1] - this->ref_positions[i * 3 + 1];
        double dz = points[i][2] - this->ref_positions[i * 3 + 2];
        double disp_sq = dx * dx + dy * dy + dz * dz;
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

    auto build_options = VesinOptions();
    build_options.cutoff = this->options.cutoff + this->options.skin;
    build_options.full = this->options.full;
    build_options.sorted = false;
    build_options.algorithm = this->options.algorithm;
    build_options.return_shifts = true;
    build_options.return_distances = false;
    build_options.return_vectors = false;
    build_options.skin = 0.0;

    this->candidates.device = {VesinCPU, 0};
    auto candidate_box = make_box_like(box, points, n_points);
    if (build_options.algorithm == VesinAutoAlgorithm && n_points >= CLUSTER_PAIR_THRESHOLD) {
        this->cluster_grid = build_cluster_grid(points, n_points, candidate_box, build_options.cutoff);
        this->cluster_candidates = build_cluster_pair_candidates(this->cluster_grid, candidate_box, build_options.cutoff);
        this->use_cluster_candidates = true;
        cpu::cluster_pair_neighbors(points, n_points, candidate_box, build_options, this->candidates);
    } else {
        size_t candidate_capacity = 0;
        cpu::stateless_neighbors(points, n_points, std::move(candidate_box), build_options, this->candidates, candidate_capacity);
    }

    this->candidate_shift_vectors.reserve(this->candidates.length);
    for (size_t k = 0; k < this->candidates.length; k++) {
        auto shift = CellShift{{
            this->candidates.shifts[k][0],
            this->candidates.shifts[k][1],
            this->candidates.shifts[k][2],
        }};
        this->candidate_shift_vectors.push_back(shift_cartesian(shift, box));
    }

    if (this->use_cluster_candidates) {
        this->simd_candidate_blocks = pack_simd_candidate_blocks(
            this->candidates,
            this->candidate_shift_vectors,
            this->simd_candidate_length
        );
    }

    this->n_points = n_points;
    this->ref_positions.resize(n_points * 3);
    std::memcpy(this->ref_positions.data(), points, n_points * 3 * sizeof(double));
    this->ref_matrix = box.matrix();
    for (size_t d = 0; d < 3; d++) {
        this->ref_periodic[d] = box.periodic(d);
    }

    this->has_cache = true;
}

void cpu::VerletState::recompute(
    const Vector* points,
    const BoundingBox& box,
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t& output_capacity
) {
    double cutoff_sq = this->options.cutoff * this->options.cutoff;

    auto initial_capacity = std::max(output_capacity, neighbors.length);

    if (this->use_cluster_candidates) {
        filter_simd_candidate_blocks(
            points,
            this->simd_candidate_blocks,
            cutoff_sq,
            options,
            neighbors,
            initial_capacity,
            output_capacity
        );
        return;
    }

    auto growable = cpu::GrowableNeighborList{neighbors, initial_capacity, options};
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

        auto vec = points[j] - points[i] + this->candidate_shift_vectors[k];
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
