#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <optional>
#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "vesin_cuda.hpp"

using namespace vesin::cuda;

// NVTX for profiling (optional, enabled if available)
#ifdef VESIN_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) \
    do {                \
    } while (0)
#define NVTX_POP() \
    do {           \
    } while (0)
#endif

#include "cuda/bruteforce.cuh"
const unsigned char CUDA_BRUTEFORCE_HEX[] = {
#include "generated/cuda/bruteforce.cu.inc"
};
const char* CUDA_BRUTEFORCE_CODE = reinterpret_cast<const char*>(CUDA_BRUTEFORCE_HEX);

#include "cuda/cell_list.cuh"
const unsigned char CUDA_CELL_LIST_HEX[] = {
#include "generated/cuda/cell_list.cu.inc"
};
const char* CUDA_CELL_LIST_CODE = reinterpret_cast<const char*>(CUDA_CELL_LIST_HEX);

#include "cuda/sort_pairs.cuh"
const unsigned char CUDA_SORT_PAIRS_HEX[] = {
#include "generated/cuda/sort_pairs.cu.inc"
};
const char* CUDA_SORT_PAIRS_CODE = reinterpret_cast<const char*>(CUDA_SORT_PAIRS_HEX);

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t DEFAULT_MAX_CELLS = 8192;
// Minimum particles per cell target for good GPU utilization.
// Lower values create more cells and reduce per-cell neighbor work, which is
// beneficial on larger systems where more coarse grids become too dense.
static constexpr size_t MIN_PARTICLES_PER_CELL = 8;

static std::optional<cudaPointerAttributes> get_ptr_attributes(const void* ptr) {
    if (ptr == nullptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        GPULITE_CUDART_CALL(cudaPointerGetAttributes(&attr, ptr));
        return attr;
    } catch (const std::runtime_error& e) {
        return std::nullopt;
    }
}

static bool is_device_ptr(const std::optional<cudaPointerAttributes>& maybe_attr, const char* name) {
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        return (attr.type == cudaMemoryTypeDevice);
    } else {
        throw std::runtime_error(
            "failed to resolve attributes for pointer: " + std::string(name)
        );
    }
}

static int32_t get_device_id(const void* ptr) {
    if (ptr == nullptr) {
        return -1;
    }

    auto maybe_attr = get_ptr_attributes(ptr);
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        if (attr.type != cudaMemoryTypeDevice) {
            return -1;
        }
        return attr.device;
    }
    return -1;
}

static size_t next_power_of_two(size_t value) {
    if (value <= 1) {
        return 1;
    }

    size_t power = 1;
    while (power < value) {
        power <<= 1;
    }

    return power;
}

static void free_output_buffers(VesinNeighborList& neighbors) {
    if ((neighbors.pairs != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.pairs), "pairs")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.pairs));
    }
    if ((neighbors.shifts != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.shifts), "shifts")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.shifts));
    }
    if ((neighbors.distances != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.distances), "distances")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.distances));
    }
    if ((neighbors.vectors != nullptr) && is_device_ptr(get_ptr_attributes(neighbors.vectors), "vectors")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    GPULITE_CUDART_CALL(cudaSetDevice(neighbors.device.device_id));

    free_output_buffers(neighbors);

    delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

void check_cuda_is_available() {
    std::string cuda_libname;
    std::string cudart_libname;
    std::string nvrtc_libname;
    std::string suggestion;
#if defined(__linux__)
    cuda_libname = "libcuda.so";
    cudart_libname = "libcudart.so(.*)";
    nvrtc_libname = "libnvrtc.so(.*)";
    suggestion = ("Try appending the directory containing this library to "
                  "your $LD_LIBRARY_PATH environment variable.");

#elif defined(_WIN32)
    cuda_libname = "nvcuda.dll";
    cudart_libname = "cudart64_*.dll";
    nvrtc_libname = "nvrtc64_*.dll";
    suggestion = ("Try adding the directory containing this library to your "
                  "system PATH, or making sure that CUDA_PATH is properly set "
                  "to your CUDA installation directory.");
#else
    cuda_libname = "cuda";
    cudart_libname = "cudart";
    nvrtc_libname = "nvrtc";
    suggestion = "Unsupported platform: unable to load CUDA libraries.";
#endif
    if (!gpulite::CUDADriver::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cuda_libname + ". " + suggestion
        );
    }

    if (!gpulite::CUDART::loaded()) {
        throw std::runtime_error(
            "Failed to load " + cudart_libname + ". " + suggestion
        );
    }

    if (!gpulite::NVRTC::loaded()) {
        throw std::runtime_error(
            "Failed to load " + nvrtc_libname + ". " + suggestion
        );
    }
}

// Reorder the output pair list (and any optional aligned outputs) so that pairs
// come out sorted by the first particle index. Both the stateless cell-list
// path and the Verlet recompute path call this so `options.sorted` has the same
// public contract on each path.
static void sort_pairs(
    VesinOptions options,
    VesinNeighborList& neighbors,
    size_t (*pairs)[2],
    int32_t (*shifts)[3],
    double* distances,
    double (*vectors)[3]
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("sort_pairs");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);
    size_t sort_capacity = next_power_of_two(neighbors.length);
    extras->sort_buffers.allocate(
        sort_capacity,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    auto* fill_kernel = factory.create<decltype(sort_pairs_fill_buffers)>(
        "sort_pairs_fill_buffers",
        CUDA_SORT_PAIRS_CODE,
        "sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    auto* bitonic_kernel = factory.create<decltype(sort_pairs_bitonic_step)>(
        "sort_pairs_bitonic_step",
        CUDA_SORT_PAIRS_CODE,
        "sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    auto* copy_back_kernel = factory.create<decltype(sort_pairs_copy_back)>(
        "sort_pairs_copy_back",
        CUDA_SORT_PAIRS_CODE,
        "sort_pairs.cu",
        {"-std=c++17", "-default-device"}
    );

    const size_t sort_threads = 256;
    const size_t sort_fill_blocks = (sort_capacity + sort_threads - 1) / sort_threads;

    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(std::max(sort_fill_blocks, static_cast<size_t>(1)));
    config.blockDim = dim3(sort_threads);

    fill_kernel->launch(
        config,
        pairs,
        shifts,
        distances,
        vectors,
        extras->sort_buffers.d_pairs_tmp,
        extras->sort_buffers.d_shifts_tmp,
        extras->sort_buffers.d_distances_tmp,
        extras->sort_buffers.d_vectors_tmp,
        neighbors.length,
        sort_capacity,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    for (size_t k = 2; k <= sort_capacity; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            bitonic_kernel->launch(
                config,
                extras->sort_buffers.d_pairs_tmp,
                extras->sort_buffers.d_shifts_tmp,
                extras->sort_buffers.d_distances_tmp,
                extras->sort_buffers.d_vectors_tmp,
                sort_capacity,
                j,
                k,
                options.return_shifts,
                options.return_distances,
                options.return_vectors
            );
        }
    }

    size_t copy_blocks = (neighbors.length + sort_threads - 1) / sort_threads;
    copy_back_kernel->launch(
        config,
        pairs,
        shifts,
        distances,
        vectors,
        extras->sort_buffers.d_pairs_tmp,
        extras->sort_buffers.d_shifts_tmp,
        extras->sort_buffers.d_distances_tmp,
        extras->sort_buffers.d_vectors_tmp,
        neighbors.length,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    NVTX_POP();
}

void vesin::cuda::allocate_output_buffers(
    VesinNeighborList& neighbors,
    size_t n_pairs,
    VesinOptions options
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras->pairs_capacity >= n_pairs && (extras->d_length_ptr != nullptr) && (extras->d_cell_check_ptr != nullptr) &&
        (extras->d_overflow_flag != nullptr) && !missing_requested_output) {
        GPULITE_CUDART_CALL(cudaMemset(extras->d_length_ptr, 0, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->d_cell_check_ptr, 0, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->d_overflow_flag, 0, sizeof(int32_t)));
    } else {
        free_output_buffers(neighbors);

        extras->pairs_capacity = n_pairs;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * n_pairs * 2));

        if (options.return_shifts) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * n_pairs * 3));
        }

        if (options.return_distances) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * n_pairs));
        }

        if (options.return_vectors) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * n_pairs * 3));
        }

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_length_ptr, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->d_length_ptr, 0, sizeof(size_t)));

        // Pinned host memory for async D2H copy
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_cell_check_ptr, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->d_cell_check_ptr, 0, sizeof(int32_t)));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_overflow_flag, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->d_overflow_flag, 0, sizeof(int32_t)));
    }
}

struct BoxChecks {
    bool use_cell_list = false;
    bool is_orthogonal = false;
};

static BoxChecks check_box(
    VesinNeighborList& neighbors,
    const double d_box[3][3],
    const bool d_periodic[3],
    size_t n_points,
    VesinOptions options
) {
    BoxChecks results;
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    if (extras->d_box_diag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_box_diag, sizeof(double) * 3));
    }
    if (extras->d_inv_box == nullptr) {
        double (*ptr)[3] = nullptr;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&ptr, sizeof(double[3][3])));
        extras->d_inv_box = ptr;
    }

    auto* box_check_kernel = factory.create<decltype(mic_box_check)>(
        "mic_box_check",
        CUDA_BRUTEFORCE_CODE,
        "bruteforce.cu",
        {"-std=c++17", "-default-device"}
    );

    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(1);
    config.blockDim = dim3(32);

    box_check_kernel->launch(
        config,
        d_box,
        d_periodic,
        options.cutoff,
        extras->d_cell_check_ptr,
        extras->d_box_diag,
        extras->d_inv_box
    );

    int32_t h_cell_check = 1;
    GPULITE_CUDART_CALL(cudaMemcpy(&h_cell_check, extras->d_cell_check_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));

    results.is_orthogonal = (h_cell_check & 2) != 0;

    double h_box_diag[3];
    GPULITE_CUDART_CALL(cudaMemcpy(h_box_diag, extras->d_box_diag, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    double min_box_dim = std::min({h_box_diag[0], h_box_diag[1], h_box_diag[2]});
    bool cutoff_requires_cell_list = options.cutoff > min_box_dim / 2.0;

    switch (options.algorithm) {
    case VesinBruteForce: {
        bool box_check_error = (h_cell_check & 1) != 0;
        if (box_check_error) {
            throw std::runtime_error("Invalid cutoff: too large for box dimensions");
        }
        results.use_cell_list = false;
        break;
    }
    case VesinCellList:
        results.use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        results.use_cell_list = cutoff_requires_cell_list || !results.is_orthogonal || n_points >= 5000;
        break;
    }

    return results;
}

static void run_cell_list(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("cell_list_total");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    // Compute effective max cells based on minimum particles per cell target
    size_t max_cells_from_particles = std::max(n_points / MIN_PARTICLES_PER_CELL, static_cast<size_t>(1));
    size_t max_cells = std::min(DEFAULT_MAX_CELLS, max_cells_from_particles);

    NVTX_PUSH("ensure_buffers");
    extras->cell_list.allocate(n_points, max_cells);
    NVTX_POP();
    auto& cl = extras->cell_list;

    // the 256 here must match the size of shared memory allocated inside the code,
    // if you update one please update the other.
    size_t THREADS_PER_BLOCK = 256;
    size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    NVTX_PUSH("kernel0_bounding_box");
    auto* bounding_kernel = factory.create<decltype(compute_bounding_box)>(
        "compute_bounding_box",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(1);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    bounding_kernel->launch(
        config,
        d_points,
        n_points,
        cl.d_face_distances,
        cl.d_bounding_min
    );
    NVTX_POP();

    NVTX_PUSH("kernel1_grid_params");
    auto* grid_kernel = factory.create<decltype(compute_cell_grid_params)>(
        "compute_cell_grid_params",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    grid_kernel->launch(
        config,
        d_box,
        d_periodic,
        options.cutoff,
        max_cells,
        extras->d_inv_box,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_n_cells_total,
        cl.d_face_distances
    );
    NVTX_POP();

    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_counts, 0, sizeof(int32_t) * max_cells));
    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_starts, 0, sizeof(int32_t) * max_cells));

    NVTX_PUSH("kernel1_assign_cells");
    auto* assign_kernel = factory.create<decltype(assign_cell_indices)>(
        "assign_cell_indices",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    config.gridDim = dim3(num_blocks_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    assign_kernel->launch(
        config,
        d_points,
        n_points,
        extras->d_inv_box,
        d_periodic,
        cl.d_n_cells,
        cl.d_face_distances,
        cl.d_bounding_min,
        cl.d_cell_indices,
        cl.d_particle_shifts
    );
    NVTX_POP();

    NVTX_PUSH("kernel2_count_particles");
    auto* count_kernel = factory.create<decltype(count_particles_per_cell)>(
        "count_particles_per_cell",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    count_kernel->launch(
        config,
        cl.d_cell_indices,
        n_points,
        cl.d_cell_counts
    );
    NVTX_POP();

    NVTX_PUSH("kernel3_prefix_sum");
    auto* prefix_kernel = factory.create<decltype(prefix_sum_cells)>(
        "prefix_sum_cells",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    size_t prefix_threads = 256;
    config.gridDim = dim3(1);
    config.blockDim = dim3(prefix_threads);

    config.dynamicSmemBytes = sizeof(int32_t) * prefix_threads;
    prefix_kernel->launch(config, cl.d_cell_counts, cl.d_cell_starts, cl.d_n_cells_total);
    config.dynamicSmemBytes = 0;

    NVTX_POP();

    NVTX_PUSH("memcpy_cell_offsets");
    GPULITE_CUDART_CALL(cudaMemcpy(
        cl.d_cell_offsets, cl.d_cell_starts, sizeof(int32_t) * max_cells, cudaMemcpyDeviceToDevice
    ));
    NVTX_POP();

    NVTX_PUSH("kernel4_scatter");
    auto* scatter_kernel = factory.create<decltype(scatter_particles)>(
        "scatter_particles",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    config.gridDim = dim3(num_blocks_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    scatter_kernel->launch(
        config,
        d_points,
        n_points,
        cl.d_cell_indices,
        cl.d_particle_shifts,
        cl.d_cell_offsets,
        cl.d_sorted_points,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices
    );
    NVTX_POP();

    NVTX_PUSH("kernel5_find_neighbors_cell_list");
    auto* find_kernel = factory.create<decltype(find_neighbors_cell_list)>(
        "find_neighbors_cell_list",
        CUDA_CELL_LIST_CODE,
        "cell_list.cu",
        {"-std=c++17", "-default-device"}
    );

    size_t THREADS_PER_PARTICLE = 8;
    size_t particles_per_block = THREADS_PER_BLOCK / THREADS_PER_PARTICLE;
    size_t num_blocks_find = (n_points + particles_per_block - 1) / particles_per_block;

    config.gridDim = dim3(num_blocks_find);
    config.blockDim = dim3(THREADS_PER_BLOCK);

    size_t max_pairs = extras->pairs_capacity;

    find_kernel->launch(
        config,
        cl.d_sorted_points,
        n_points,
        d_box,
        d_periodic,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices,
        cl.d_cell_starts,
        cl.d_cell_counts,
        options.cutoff,
        options.full,
        options.return_shifts,
        options.return_distances,
        options.return_vectors,
        extras->d_length_ptr,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        max_pairs,
        extras->d_overflow_flag
    );
    NVTX_POP();

    NVTX_POP(); // cell_list_total
}

static void run_brute_force(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors,
    bool is_orthogonal
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("brute_force_total");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    size_t THREADS_PER_BLOCK = 128;
    double cutoff2 = options.cutoff * options.cutoff;

    size_t num_half_pairs = n_points * (n_points - 1) / 2;

    size_t max_pairs = extras->pairs_capacity;

    auto config = gpulite::LaunchConfig();
    size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    config.gridDim = dim3(std::max(num_blocks, static_cast<size_t>(1)));
    config.blockDim = dim3(THREADS_PER_BLOCK);

    if (is_orthogonal) {
        if (options.full) {
            NVTX_PUSH("brute_force_full_orthogonal");
            auto* kernel = factory.create<decltype(brute_force_full_orthogonal)>(
                "brute_force_full_orthogonal",
                CUDA_BRUTEFORCE_CODE,
                "bruteforce.cu",
                {"-std=c++17", "-default-device"}
            );

            kernel->launch(
                config,
                d_points,
                n_points,
                extras->d_box_diag,
                d_periodic,
                cutoff2,
                options.return_shifts,
                options.return_distances,
                options.return_vectors,
                extras->d_length_ptr,
                neighbors.pairs,
                neighbors.shifts,
                neighbors.distances,
                neighbors.vectors,
                max_pairs,
                extras->d_overflow_flag
            );
            NVTX_POP();
        } else {
            NVTX_PUSH("brute_force_half_orthogonal");
            auto* kernel = factory.create<decltype(brute_force_half_orthogonal)>(
                "brute_force_half_orthogonal",
                CUDA_BRUTEFORCE_CODE,
                "bruteforce.cu",
                {"-std=c++17", "-default-device"}
            );

            kernel->launch(
                config,
                d_points,
                n_points,
                extras->d_box_diag,
                d_periodic,
                cutoff2,
                options.return_shifts,
                options.return_distances,
                options.return_vectors,
                extras->d_length_ptr,
                neighbors.pairs,
                neighbors.shifts,
                neighbors.distances,
                neighbors.vectors,
                max_pairs,
                extras->d_overflow_flag
            );
            NVTX_POP();
        }
    } else {
        if (options.full) {
            NVTX_PUSH("brute_force_full_general");
            auto* kernel = factory.create<decltype(brute_force_full_general)>(
                "brute_force_full_general",
                CUDA_BRUTEFORCE_CODE,
                "bruteforce.cu",
                {"-std=c++17", "-default-device"}
            );

            kernel->launch(
                config,
                d_points,
                n_points,
                d_box,
                extras->d_inv_box,
                d_periodic,
                cutoff2,
                options.return_shifts,
                options.return_distances,
                options.return_vectors,
                extras->d_length_ptr,
                neighbors.pairs,
                neighbors.shifts,
                neighbors.distances,
                neighbors.vectors,
                max_pairs,
                extras->d_overflow_flag
            );
            NVTX_POP();
        } else {
            NVTX_PUSH("brute_force_half_general");
            auto* kernel = factory.create<decltype(brute_force_half_general)>(
                "brute_force_half_general",
                CUDA_BRUTEFORCE_CODE,
                "bruteforce.cu",
                {"-std=c++17", "-default-device"}
            );

            kernel->launch(
                config,
                d_points,
                n_points,
                d_box,
                extras->d_inv_box,
                d_periodic,
                cutoff2,
                options.return_shifts,
                options.return_distances,
                options.return_vectors,
                extras->d_length_ptr,
                neighbors.pairs,
                neighbors.shifts,
                neighbors.distances,
                neighbors.vectors,
                max_pairs,
                extras->d_overflow_flag
            );
            NVTX_POP();
        }
    }

    NVTX_POP(); // brute_force_total
}

static void finalize_output(
    size_t n_points,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("async_copy_and_sync");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        extras->pinned_length_ptr,
        extras->d_length_ptr,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));

    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    neighbors.length = *extras->pinned_length_ptr;

    NVTX_POP();

    if (options.sorted && neighbors.length > 1) {
        sort_pairs(
            options,
            neighbors,
            neighbors.pairs,
            neighbors.shifts,
            neighbors.distances,
            neighbors.vectors
        );
    }
}

void vesin::cuda::neighbors(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(neighbors.device.type == VesinCUDA);
    check_cuda_is_available();

    // check that all pointers are are device pointers
    if (!is_device_ptr(get_ptr_attributes(d_points), "points")) {
        throw std::runtime_error("`points` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(get_ptr_attributes(d_box), "box")) {
        throw std::runtime_error("`box` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(get_ptr_attributes(d_periodic), "periodic")) {
        throw std::runtime_error("`periodic` pointer is not allocated on a CUDA device");
    }

    auto device_id = get_device_id(d_points);
    if (device_id != get_device_id(d_box)) {
        throw std::runtime_error("`points` and `box` do not exist on the same device");
    }

    if (device_id != get_device_id(d_periodic)) {
        throw std::runtime_error("`points` and `periodic` do not exist on the same device");
    }

    if (device_id != neighbors.device.device_id) {
        throw std::runtime_error("`points`, `box` and `periodic` device differs from input neighbors device_id");
    }

    auto* extras = vesin::cuda::get_cuda_extras(neighbors);
    if (options.skin > 0.0) {
        extras->verlet_cache.run(
            d_points, n_points, d_box, d_periodic, options, neighbors
        );

    } else {
        // no skin, free any remaining verlet buffers from a previous call
        extras->verlet_cache = VerletCache();

        size_t max_pairs_per_point = std::max(
            static_cast<size_t>(VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT),
            static_cast<size_t>(std::ceil(std::pow(options.cutoff, 3)))
        );

        auto* env_max_pairs = std::getenv("VESIN_CUDA_MAX_PAIRS_PER_POINT");
        if (env_max_pairs != nullptr) {
            auto length = std::strlen(env_max_pairs);
            char* end = nullptr;
            errno = 0;
            auto parsed_max_pairs_per_point = std::strtoll(env_max_pairs, &end, 10);
            if (errno != 0 || end != env_max_pairs + length || parsed_max_pairs_per_point <= 0) {
                throw std::runtime_error(
                    "Invalid value for VESIN_CUDA_MAX_PAIRS_PER_POINT: '" +
                    std::string(env_max_pairs) + "'"
                );
            }
            max_pairs_per_point = static_cast<size_t>(parsed_max_pairs_per_point);
        }

        vesin::cuda::allocate_output_buffers(neighbors, n_points * max_pairs_per_point, options);

        auto box_checks = check_box(neighbors, d_box, d_periodic, n_points, options);
        if (box_checks.use_cell_list) {
            run_cell_list(
                d_points, n_points, d_box, d_periodic, options, neighbors
            );
        } else {
            run_brute_force(
                d_points, n_points, d_box, d_periodic, options, neighbors, box_checks.is_orthogonal
            );
        }

        GPULITE_CUDART_CALL(cudaDeviceSynchronize());

        // Check for overflow
        int h_overflow_flag = 0;
        GPULITE_CUDART_CALL(cudaMemcpy(
            &h_overflow_flag,
            extras->d_overflow_flag,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        if (h_overflow_flag != 0) {
            throw std::runtime_error(
                "The number of neighbor pairs exceeds the maximum capacity of " +
                std::to_string(extras->pairs_capacity) + " (max_pairs_per_point=" +
                std::to_string(max_pairs_per_point) + "; n_points=" +
                std::to_string(n_points) + "). " +
                "Consider reducing the cutoff distance, or explicitly setting " +
                "VESIN_CUDA_MAX_PAIRS_PER_POINT as an environment variable."
            );
        }
    }

    finalize_output(n_points, options, neighbors);
}
