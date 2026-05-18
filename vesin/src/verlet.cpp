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

// Vectorized gather of pair deltas using Highway for the Verlet recompute hot path.
// Gathers x/y/z for atom i and j using byte-offset gathers, computes (j - i) + shift
// in SIMD registers, and stores to the dx/dy/dz temporaries for the subsequent filter.
HWY_ATTR
static void gather_pair_deltas(
    const Vector* HWY_RESTRICT points,
    const size_t* HWY_RESTRICT first,
    const size_t* HWY_RESTRICT second,
    const double* HWY_RESTRICT shift_x,
    const double* HWY_RESTRICT shift_y,
    const double* HWY_RESTRICT shift_z,
    size_t count,
    double* HWY_RESTRICT dx,
    double* HWY_RESTRICT dy,
    double* HWY_RESTRICT dz
) {
    const hn::ScalableTag<double> d;
    const hn::ScalableTag<int64_t> di64;
    const size_t N = hn::Lanes(d);
    const double* base = reinterpret_cast<const double*>(points);
    const auto k_three = hn::Set(di64, int64_t{3});
    const auto k_one = hn::Set(di64, int64_t{1});

    size_t lane = 0;
    for (; lane + N <= count; lane += N) {
        // Load atom indices as signed (safe for practical atom counts < 2^63)
        // Note: first/second are size_t but values are non-negative.
        alignas(64) int64_t i_idx[8] = {};
        alignas(64) int64_t j_idx[8] = {};
        for (size_t k = 0; k < N; k++) {
            i_idx[k] = static_cast<int64_t>(first[lane + k]);
            j_idx[k] = static_cast<int64_t>(second[lane + k]);
        }
        auto v_i = hn::Load(di64, i_idx);
        auto v_j = hn::Load(di64, j_idx);

        // Element indices for x: atom * 3
        auto idx_i_x = hn::Mul(v_i, k_three);
        auto idx_j_x = hn::Mul(v_j, k_three);

        auto vxi = hn::GatherIndex(d, base, idx_i_x);
        auto vxj = hn::GatherIndex(d, base, idx_j_x);

        // y: +1
        auto idx_i_y = hn::Add(idx_i_x, k_one);
        auto idx_j_y = hn::Add(idx_j_x, k_one);
        auto vyi = hn::GatherIndex(d, base, idx_i_y);
        auto vyj = hn::GatherIndex(d, base, idx_j_y);

        // z: +2
        auto idx_i_z = hn::Add(idx_i_y, k_one);
        auto idx_j_z = hn::Add(idx_j_y, k_one);
        auto vzi = hn::GatherIndex(d, base, idx_i_z);
        auto vzj = hn::GatherIndex(d, base, idx_j_z);

        // deltas j - i
        auto vdx = hn::Sub(vxj, vxi);
        auto vdy = hn::Sub(vyj, vyi);
        auto vdz = hn::Sub(vzj, vzi);

        // load and add precomputed Cartesian shifts
        auto vsx = hn::Load(d, shift_x + lane);
        auto vsy = hn::Load(d, shift_y + lane);
        auto vsz = hn::Load(d, shift_z + lane);

        vdx = hn::Add(vdx, vsx);
        vdy = hn::Add(vdy, vsy);
        vdz = hn::Add(vdz, vsz);

        hn::Store(vdx, d, dx + lane);
        hn::Store(vdy, d, dy + lane);
        hn::Store(vdz, d, dz + lane);
    }

    // Scalar tail for remainder (when count % N != 0)
    for (; lane < count; lane++) {
        auto i = first[lane];
        auto j = second[lane];
        dx[lane] = points[j][0] - points[i][0] + shift_x[lane];
        dy[lane] = points[j][1] - points[i][1] + shift_y[lane];
        dz[lane] = points[j][2] - points[i][2] + shift_z[lane];
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
    alignas(64) double dist[CLUSTER_SIZE_CPU];
    uint8_t mask[CLUSTER_SIZE_CPU];
    const bool need_distance = options.return_distances;

    for (const auto& block : blocks) {
        // Vectorized gather + arithmetic for the valid lanes (uses Highway Gather
        // for random-access position loads, enabling better load parallelism on AVX2/AVX-512).
        if (block.count > 0) {
            gather_pair_deltas(
                points,
                block.first,
                block.second,
                block.shift_x,
                block.shift_y,
                block.shift_z,
                block.count,
                dx,
                dy,
                dz
            );
        }
        for (size_t lane = block.count; lane < CLUSTER_SIZE_CPU; lane++) {
            dx[lane] = std::numeric_limits<double>::infinity();
            dy[lane] = 0.0;
            dz[lane] = 0.0;
        }

        simd_filter_deltas(dx, dy, dz, cutoff_sq, dist_sq, mask);

        // SIMD sqrt: precompute once for the whole block instead of one
        // scalar std::sqrt per kept pair in the lane loop. Highway picks
        // the widest available vector lane (AVX-512: 8 doubles, AVX2: 4).
        // Wasted lanes for filtered-out pairs are cheap relative to the
        // scalar loop overhead. Only run when distances are requested.
        if (need_distance) {
            const hn::ScalableTag<double> d;
            const size_t N = hn::Lanes(d);
            for (size_t lane = 0; lane < CLUSTER_SIZE_CPU; lane += N) {
                hn::Store(hn::Sqrt(hn::Load(d, dist_sq + lane)), d, dist + lane);
            }
        }

        // Pre-grow once per block. Worst case is every lane in this block
        // passes the filter, so reserve growable.length() + block.count up
        // front. This hoists the per-pair capacity branch out of the hot
        // lane loop below (cachegrind: 34% of total instructions were in
        // the per-pair set_*'s capacity check + grow); the unchecked
        // set_*_unchecked variants below skip it entirely.
        growable.ensure_capacity(growable.length() + block.count);

        for (size_t lane = 0; lane < block.count; lane++) {
            if (!mask[lane]) {
                continue;
            }

            auto index = growable.length();
            growable.set_pair_unchecked(index, block.first[lane], block.second[lane]);

            if (options.return_shifts) {
                growable.set_shift_unchecked(index, block.shifts[lane]);
            }

            if (need_distance) {
                growable.set_distance_unchecked(index, dist[lane]);
            }

            if (options.return_vectors) {
                growable.set_vector_unchecked(index, Vector{dx[lane], dy[lane], dz[lane]});
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

    // Pre-grow once for the worst case (every candidate passes the filter)
    // so the inner loop never branches on capacity. Mirror change to the
    // SIMD-block path above.
    growable.ensure_capacity(this->candidates.length);

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
            growable.set_pair_unchecked(idx, i, j);

            if (options.return_shifts) {
                growable.set_shift_unchecked(idx, shift);
            }

            if (options.return_distances) {
                growable.set_distance_unchecked(idx, std::sqrt(dist_sq));
            }

            if (options.return_vectors) {
                growable.set_vector_unchecked(idx, vec);
            }

            growable.increment_length();
        }
    }

    if (options.sorted) {
        growable.sort();
    }

    output_capacity = growable.capacity;
}
