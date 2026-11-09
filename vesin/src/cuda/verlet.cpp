#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "./extra_data.hpp"
#include "./neighbors.hpp"
#include "./padding.hpp"
#include "./sort_pairs.hpp"
#include "./verlet.hpp"

using namespace vesin::cuda;

#include "kernels/verlet.cuh"
const uint8_t CUDA_VERLET_HEX[] = {
#include "generated/kernels/verlet.cu.inc"
};
static const char* CUDA_VERLET_CODE = reinterpret_cast<const char*>(CUDA_VERLET_HEX);

static double corner_point_displacements(
    const double h_box[9],
    const double h_ref_box[9]
) {
    double delta1 = 0.0;
    double delta2 = 0.0;
    for (size_t i = 0; i < 8; i++) {
        const double b0 = ((i & 1) != 0) ? 1.0 : 0.0;
        const double b1 = ((i & 2) != 0) ? 1.0 : 0.0;
        const double b2 = ((i & 4) != 0) ? 1.0 : 0.0;
        double delta = 0.0;
        for (size_t j = 0; j < 3; j++) {
            double c = h_box[j] * b0 + h_box[j + 3] * b1 + h_box[j + 6] * b2;
            double c_ref = h_ref_box[j] * b0 + h_ref_box[j + 3] * b1 + h_ref_box[j + 6] * b2;
            double del = c - c_ref;
            delta += del * del;
        }
        if (delta > delta1) {
            delta2 = delta1;
            delta1 = delta;
        } else if (delta > delta2) {
            delta2 = delta;
        }
    }
    return std::sqrt(delta1) + std::sqrt(delta2);
}

VerletCache::~VerletCache() {
    // The work using these buffers must be finished at this point, see
    // `vesin::cuda::free_neighbors`
    try {
        if (this->candidates_.device.type == VesinCUDA) {
            vesin::cuda::free_neighbors(this->candidates_);
        }
        GPULITE_CUDART_CALL(cudaFree(this->d_ref_points_));
        GPULITE_CUDART_CALL(cudaFree(this->d_rebuild_flag_));
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing VerletCache buffers: " << e.what() << std::endl;
    }
}

VerletCache::VerletCache(VerletCache&& other) noexcept:
    VerletCache() {
    *this = std::move(other);
}

VerletCache& VerletCache::operator=(VerletCache&& other) noexcept {
    if (this != &other) {
        this->~VerletCache();

        this->ref_points_capacity_ = other.ref_points_capacity_;
        other.ref_points_capacity_ = 0;

        this->d_ref_points_ = other.d_ref_points_;
        other.d_ref_points_ = nullptr;

        this->n_ref_points_ = other.n_ref_points_;
        other.n_ref_points_ = 0;

        std::memcpy(this->ref_box_, other.ref_box_, sizeof(double) * 9);
        std::memset(other.ref_box_, 0, sizeof(double) * 9);

        std::memcpy(this->ref_periodic_, other.ref_periodic_, sizeof(bool) * 3);
        std::memset(other.ref_periodic_, 0, sizeof(bool) * 3);

        this->options_ = other.options_;
        other.options_ = VesinOptions();

        this->has_cache_ = other.has_cache_;
        other.has_cache_ = false;

        this->candidates_ = other.candidates_;
        other.candidates_ = VesinNeighborList();

        this->d_rebuild_flag_ = other.d_rebuild_flag_;
        other.d_rebuild_flag_ = nullptr;
    }
    return *this;
}

void VerletCache::free_buffers(cudaStream_t stream) {
    if (this->candidates_.device.type == VesinCUDA) {
        vesin::cuda::free_neighbors(this->candidates_);
    }

    this->candidates_ = VesinNeighborList();
    GPULITE_CUDART_CALL(cudaFreeAsync(this->d_ref_points_, stream));
    GPULITE_CUDART_CALL(cudaFreeAsync(this->d_rebuild_flag_, stream));

    this->d_ref_points_ = nullptr;
    this->d_rebuild_flag_ = nullptr;
    this->ref_points_capacity_ = 0;
    this->n_ref_points_ = 0;
    this->options_ = VesinOptions();
    this->has_cache_ = false;
}

bool VerletCache::options_changed(VesinOptions options) const {
    return this->options_.cutoff != options.cutoff ||
           this->options_.skin != options.skin ||
           this->options_.full != options.full;
}

bool VerletCache::box_size_changed(const double h_box[9]) const {
    for (size_t i = 0; i < 9; i++) {
        if (std::abs(this->ref_box_[i] - h_box[i]) > 1e-12) {
            return true;
        }
    }

    return false;
}

bool VerletCache::box_periodic_changed(const bool h_periodic[3]) const {
    for (size_t i = 0; i < 3; i++) {
        if (this->ref_periodic_[i] != h_periodic[i]) {
            return true;
        }
    }

    return false;
}

void VerletCache::allocate_ref_buffers(size_t n_points, cudaStream_t stream) {
    if (this->ref_points_capacity_ < n_points) {
        GPULITE_CUDART_CALL(cudaFreeAsync(this->d_ref_points_, stream));
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&this->d_ref_points_, sizeof(double) * n_points * 3, stream));
        this->ref_points_capacity_ = n_points;
    }

    if (this->d_rebuild_flag_ == nullptr) {
        GPULITE_CUDART_CALL(cudaMallocAsync((void**)&this->d_rebuild_flag_, sizeof(int32_t), stream));
    }
}

void VerletCache::rebuild_cache(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto device_id = neighbors.device.device_id;
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto* pinned = extras->pinned;

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        pinned->box, d_box, sizeof(double) * 9, cudaMemcpyDeviceToHost, extras->stream
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        pinned->periodic, d_periodic, sizeof(bool) * 3, cudaMemcpyDeviceToHost, extras->stream
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(extras->stream));

    const double* h_box = pinned->box;
    const bool* h_periodic = pinned->periodic;

    bool can_reuse = this->has_cache_ && this->n_ref_points_ == n_points && !this->box_periodic_changed(h_periodic);

    // When the box changed (e.g. NPT), part of the skin budget is eaten by the
    // affine box deformation. Shrink the per-point displacement threshold by
    // half the corner displacement, and force a rebuild when the box moved too
    // much. The corner displacement uses the full box matrix: non-periodic
    // directions carry no shift, so including them only over-estimates the
    // bound (more rebuilds, never fewer), keeping any mix of periodicity correct.
    double corner_displacement = 0.0;
    if (can_reuse && this->box_size_changed(h_box)) {
        corner_displacement = corner_point_displacements(h_box, this->ref_box_);
        if (corner_displacement >= options.skin) {
            can_reuse = false;
        }
    }

    if (can_reuse) {
        auto& factory = gpulite::KernelFactory::instance(device_id);

        GPULITE_CUDART_CALL(cudaMemsetAsync(this->d_rebuild_flag_, 0, sizeof(int32_t), extras->stream));

        auto* kernel = factory.create<decltype(check_verlet_displacements)>(
            "check_verlet_displacements",
            CUDA_VERLET_CODE,
            "verlet.cu",
            {"-std=c++17", "-default-device"}
        );

        size_t threads = 256;
        size_t blocks = (n_points + threads - 1) / threads;

        auto config = gpulite::LaunchConfig();
        config.stream = extras->stream;
        config.gridDim = dim3(std::max(blocks, static_cast<size_t>(1)));
        config.blockDim = dim3(threads);

        double half_threshold = (options.skin - corner_displacement) / 2.0;
        double half_skin_sq = half_threshold * half_threshold;

        kernel->launch(
            config,
            d_points,
            this->d_ref_points_,
            n_points,
            half_skin_sq,
            this->d_rebuild_flag_
        );

        GPULITE_CUDART_CALL(cudaMemcpyAsync(
            &pinned->rebuild, this->d_rebuild_flag_,
            sizeof(int32_t), cudaMemcpyDeviceToHost, extras->stream
        ));
        GPULITE_CUDART_CALL(cudaStreamSynchronize(extras->stream));

        if (pinned->rebuild == 0) {
            // Cache is still valid, no need to rebuild
            return;
        }
    }

    if (this->candidates_.device.type == VesinCUDA) {
        vesin::cuda::free_neighbors(this->candidates_);
    }

    this->candidates_ = VesinNeighborList();
    this->candidates_.device = {VesinCUDA, device_id};

    auto build_options = VesinOptions();
    build_options.cutoff = options.cutoff + options.skin;
    build_options.full = options.full;
    build_options.sorted = false;
    build_options.algorithm = VesinCellList;
    build_options.return_shifts = true;
    build_options.return_distances = false;
    build_options.return_vectors = false;
    build_options.skin = 0.0;

    vesin::cuda::neighbors(
        d_points,
        n_points,
        d_box,
        d_periodic,
        extras->stream,
        build_options,
        this->candidates_
    );

    this->allocate_ref_buffers(n_points, extras->stream);
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        this->d_ref_points_,
        d_points,
        sizeof(double) * n_points * 3,
        cudaMemcpyDeviceToDevice,
        extras->stream
    ));

    std::memcpy(this->ref_box_, h_box, sizeof(double) * 9);
    std::memcpy(this->ref_periodic_, h_periodic, sizeof(bool) * 3);
    this->n_ref_points_ = n_points;
    this->options_ = options;
    this->has_cache_ = true;
}

void VerletCache::filter_neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) const {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    if (options.padded_to > 0) {
        vesin::cuda::prepare_padding(neighbors, n_points, options);
    } else if (!extras->pre_allocated_outputs) {
        // the number of candidates is an upper bound on the number of pairs
        vesin::cuda::allocate_output_buffers(neighbors, this->candidates_.length, options);
    }

    // The kernel below accumulates the number of pairs it kept with atomics, so
    // the counter has to start from zero on every call.
    GPULITE_CUDART_CALL(cudaMemsetAsync(extras->d_length_ptr, 0, sizeof(size_t), extras->stream));
    GPULITE_CUDART_CALL(cudaMemsetAsync(extras->d_overflow_flag, 0, sizeof(int32_t), extras->stream));

    size_t threads = 256;
    size_t blocks = (this->candidates_.length + threads - 1) / threads;
    auto* kernel = factory.create<decltype(filter_verlet_candidates)>(
        "filter_verlet_candidates",
        CUDA_VERLET_CODE,
        "verlet.cu",
        {"-std=c++17", "-default-device"}
    );

    auto config = gpulite::LaunchConfig();
    config.stream = extras->stream;
    config.gridDim = dim3(std::max(blocks, static_cast<size_t>(1)));
    config.blockDim = dim3(threads);

    kernel->launch(
        config,
        d_points,
        d_box,
        options.cutoff,
        options.return_shifts,
        options.return_distances,
        options.return_vectors,
        this->candidates_.pairs,
        this->candidates_.shifts,
        this->candidates_.length,
        extras->d_length_ptr,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        extras->pairs_capacity,
        extras->d_overflow_flag,
        static_cast<int32_t>(options.padded_to),
        extras->d_pairs_per_point
    );
}

void VerletCache::run(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    if (this->options_changed(options)) {
        auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
        this->free_buffers(extras->stream);
    }

    this->rebuild_cache(d_points, n_points, d_box, d_periodic, options, neighbors);
    this->filter_neighbors(d_points, n_points, d_box, options, neighbors);

    if (options.padded_to > 0) {
        // the output is sorted by construction when padding
        vesin::cuda::finish_padding(neighbors, n_points, options);
        return;
    }

    // Read back the pair count from the GPU-side atomic counter, and the
    // overflow flag.
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->length,
        extras->d_length_ptr,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        extras->stream
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->overflow,
        extras->d_overflow_flag,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        extras->stream
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(extras->stream));

    neighbors.length = extras->pinned->length;

    if (extras->pinned->overflow != 0) {
        // vesin-allocated outputs can hold all the candidates, so this only
        // happens with pre-allocated outputs
        assert(extras->pre_allocated_outputs);
        throw vesin::pre_allocated_too_small(extras->pairs_capacity, neighbors.length);
    }

    // Verlet output is unsorted; sort if requested
    if (options.sorted && neighbors.length > 1) {
        vesin::cuda::sort_pairs(n_points, options, neighbors);
    }
}
