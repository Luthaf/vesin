#include "vesin_cuda.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "cuda_cache.hpp"
#include "dynamic_cuda.hpp"

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

using namespace vesin::cuda;
using namespace std;

// Debug timing: set VESIN_DEBUG_TIMING=1 to enable per-kernel timing with sync
static bool debug_timing_enabled() {
    // Check every time (not cached) to allow runtime enable/disable
    const char* env = std::getenv("VESIN_DEBUG_TIMING");
    return (env && (std::string(env) == "1" || std::string(env) == "true"));
}

// Sync and print timing if debug enabled
#define DEBUG_SYNC_AND_TIME(name)                                                                         \
    if (debug_timing_enabled()) {                                                                         \
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize());                                        \
        auto now = std::chrono::high_resolution_clock::now();                                             \
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - _debug_start).count(); \
        fprintf(stderr, "[TIMING] %s: %ld us\n", name, elapsed);                                          \
        fflush(stderr);                                                                                   \
        _debug_start = now;                                                                               \
    }

#define DEBUG_TIMING_START()                                       \
    auto _debug_start = std::chrono::high_resolution_clock::now(); \
    if (debug_timing_enabled()) {                                  \
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize()); \
        _debug_start = std::chrono::high_resolution_clock::now();  \
    }

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t MAX_CELLS = 8192;
// Minimum particles per cell target for good GPU utilization
// Higher values = fewer cells = more work per block but larger search range
static constexpr int MIN_PARTICLES_PER_CELL = 128;

static std::optional<cudaPointerAttributes> getPtrAttributes(const void* ptr) {
    if (!ptr) {
        return std::nullopt;
    }

    try {
        cudaPointerAttributes attr;
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaPointerGetAttributes(&attr, ptr));
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

static int get_device_id(const void* ptr) {
    if (!ptr) {
        return -1;
    }
    auto maybe_attr = getPtrAttributes(ptr);
    if (maybe_attr) {
        const cudaPointerAttributes& attr = *maybe_attr;
        if (attr.type != cudaMemoryTypeDevice) {
            return -1;
        }
        return attr.device;
    }
    return -1;
}

static void free_cell_list_buffers(CellListBuffers& cl) {
    if (cl.cell_indices) {
        CUDART_INSTANCE.cudaFree(cl.cell_indices);
    }
    if (cl.particle_shifts) {
        CUDART_INSTANCE.cudaFree(cl.particle_shifts);
    }
    if (cl.cell_counts) {
        CUDART_INSTANCE.cudaFree(cl.cell_counts);
    }
    if (cl.cell_starts) {
        CUDART_INSTANCE.cudaFree(cl.cell_starts);
    }
    if (cl.cell_offsets) {
        CUDART_INSTANCE.cudaFree(cl.cell_offsets);
    }
    if (cl.sorted_positions) {
        CUDART_INSTANCE.cudaFree(cl.sorted_positions);
    }
    if (cl.sorted_indices) {
        CUDART_INSTANCE.cudaFree(cl.sorted_indices);
    }
    if (cl.sorted_shifts) {
        CUDART_INSTANCE.cudaFree(cl.sorted_shifts);
    }
    if (cl.sorted_cell_indices) {
        CUDART_INSTANCE.cudaFree(cl.sorted_cell_indices);
    }
    if (cl.inv_box) {
        CUDART_INSTANCE.cudaFree(cl.inv_box);
    }
    if (cl.n_cells) {
        CUDART_INSTANCE.cudaFree(cl.n_cells);
    }
    if (cl.n_search) {
        CUDART_INSTANCE.cudaFree(cl.n_search);
    }
    if (cl.n_cells_total) {
        CUDART_INSTANCE.cudaFree(cl.n_cells_total);
    }
    cl = CellListBuffers();
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    if (this->length_ptr) {
        CUDART_INSTANCE.cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr) {
        CUDART_INSTANCE.cudaFree(this->cell_check_ptr);
    }
    free_cell_list_buffers(this->cell_list);
}

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (!neighbors->opaque) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        vesin::cuda::CudaNeighborListExtras* test =
            static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

static void reset(VesinNeighborList& neighbors) {

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    if (neighbors.pairs && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.pairs));
    }
    if (neighbors.shifts && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.shifts));
    }
    if (neighbors.distances && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.distances));
    }
    if (neighbors.vectors && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    extras->length_ptr = nullptr;

    // Free pinned memory if allocated
    if (extras->pinned_length_ptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFreeHost(extras->pinned_length_ptr));
        extras->pinned_length_ptr = nullptr;
    }

    free_cell_list_buffers(extras->cell_list);

    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {

    assert(neighbors.device.type == VesinCUDA);

    int curr_device = -1;
    int device_id = -1;

    if (neighbors.pairs) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if (device_id && curr_device != device_id) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if (device_id && curr_device != device_id) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(curr_device));
    }

    if (neighbors.opaque) {
        delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
        neighbors.opaque = nullptr;
    }
}

void checkCuda() {
    if (!CUDA_DRIVER_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libcuda.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!CUDART_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libcudart.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }

    if (!NVRTC_INSTANCE.loaded()) {
        throw std::runtime_error(
            "Failed to load libnvrtc.so. Try appending the directory containing this library to "
            "your $LD_LIBRARY_PATH environment variable."
        );
    }
}

// Ensure cell list buffers are allocated with sufficient capacity
static void ensure_cell_list_buffers(
    CellListBuffers& cl,
    size_t n_points,
    size_t n_cells_total
) {
    bool need_realloc_points = (cl.max_points < n_points);
    bool need_realloc_cells = (cl.max_cells < n_cells_total);

    if (need_realloc_points) {
        // Free old point-related buffers
        if (cl.cell_indices) {
            CUDART_INSTANCE.cudaFree(cl.cell_indices);
        }
        if (cl.particle_shifts) {
            CUDART_INSTANCE.cudaFree(cl.particle_shifts);
        }
        if (cl.sorted_positions) {
            CUDART_INSTANCE.cudaFree(cl.sorted_positions);
        }
        if (cl.sorted_indices) {
            CUDART_INSTANCE.cudaFree(cl.sorted_indices);
        }
        if (cl.sorted_shifts) {
            CUDART_INSTANCE.cudaFree(cl.sorted_shifts);
        }
        if (cl.sorted_cell_indices) {
            CUDART_INSTANCE.cudaFree(cl.sorted_cell_indices);
        }

        size_t new_max = static_cast<size_t>(1.2 * n_points);
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_indices, sizeof(int) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.particle_shifts, sizeof(int32_t) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_positions, sizeof(double) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_indices, sizeof(int) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_shifts, sizeof(int32_t) * new_max * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.sorted_cell_indices, sizeof(int) * new_max));
        cl.max_points = new_max;
    }

    if (need_realloc_cells) {
        // Free old cell-related buffers
        if (cl.cell_counts) {
            CUDART_INSTANCE.cudaFree(cl.cell_counts);
        }
        if (cl.cell_starts) {
            CUDART_INSTANCE.cudaFree(cl.cell_starts);
        }
        if (cl.cell_offsets) {
            CUDART_INSTANCE.cudaFree(cl.cell_offsets);
        }

        size_t new_max = static_cast<size_t>(1.2 * n_cells_total);
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_counts, sizeof(int) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_starts, sizeof(int) * new_max));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.cell_offsets, sizeof(int) * new_max));
        cl.max_cells = new_max;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (!cl.inv_box) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.inv_box, sizeof(double) * 9));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_cells, sizeof(int) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_search, sizeof(int) * 3));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&cl.n_cells_total, sizeof(int)));
    }
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto _function_start = std::chrono::high_resolution_clock::now();

    static const char* CUDA_CODE =
#include "generated/mic_neighbourlist.cu"
        ;

    static const char* CELL_LIST_CODE =
#include "generated/cell_list.cu"
        ;

    assert(neighbors.device.type == VesinCUDA);
    assert(!options.sorted && "Sorting is not supported in CUDA version of Vesin");

    // Check if CUDA is available
    checkCuda();

    // assert both points and box are device pointers
    assert(is_device_ptr(getPtrAttributes(points)) && "points pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(box)) && "box pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(periodic)) && "periodic pointer is not allocated on a CUDA device");

    int device = get_device_id(points);
    // assert both points and box are on the same device
    assert((device == get_device_id(box)) && "`points` and `box` do not exist on the same device");
    assert((device == get_device_id(periodic)) && "`points` and `periodic` do not exist on the same device");
    assert((device == neighbors.device.device_id) && "`points`, `box` and `periodic` device differs from input neighbors device_id");

    auto _alloc_start = std::chrono::high_resolution_clock::now();
    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

    // if allocated_device is different from the input device, we need to reset
    if (extras->allocated_device != device) {
        // first switch to previous device
        if (extras->allocated_device >= 0) {
            CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(extras->allocated_device));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaSetDevice(device));
        extras->allocated_device = device;
    }

    // make sure the allocations can fit n_points
    if (extras->capacity >= n_points &&
        extras->length_ptr) {
        // allocation fits, reset the counters
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));
    } else {
        // need a new allocation, so reset and reallocate
        reset(neighbors);
        auto max_pairs = static_cast<size_t>(1.2 * n_points * VESIN_CUDA_MAX_PAIRS_PER_POINT);

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * max_pairs * 2));

        if (options.return_shifts) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * max_pairs * 3)
            );
        }

        if (options.return_distances) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.distances, sizeof(double) * max_pairs)
            );
        }

        if (options.return_vectors) {
            CUDART_SAFE_CALL(
                CUDART_INSTANCE.cudaMalloc((void**)&neighbors.vectors, sizeof(double) * max_pairs * 3)
            );
        }

        // Allocate device memory for the length counter
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        // Allocate pinned host memory for fast D2H copy
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaHostAlloc(
            (void**)&extras->pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));

        extras->capacity = static_cast<size_t>(1.2 * n_points);
    }

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_box = reinterpret_cast<const double*>(box);
    const bool* d_periodic = periodic;

    size_t* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    int* d_shifts = reinterpret_cast<int*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    size_t* d_pair_counter = extras->length_ptr;
    int* d_cell_check = extras->cell_check_ptr;

    // Get or create kernel factory
    auto& factory = KernelFactory::instance();

    if (debug_timing_enabled()) {
        // Time from function entry to here (allocation + setup)
        // NOTE: Not syncing here - we want to measure CPU-side setup only
        auto alloc_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::high_resolution_clock::now() - _alloc_start
        )
                                 .count();
        fprintf(stderr, "[TIMING] allocation+setup (no sync): %ld us\n", alloc_elapsed);
        fflush(stderr);
    }

    // First check box dimensions with mic_box_check kernel
    auto _box_check_start = std::chrono::high_resolution_clock::now();

    auto* box_check_kernel = factory.create(
        "mic_box_check",
        CUDA_CODE,
        "mic_neighbourlist.cu",
        {"-std=c++17"}
    );

    // Prepare arguments for box check kernel
    std::vector<void*> box_check_args = {&d_box, &d_periodic, &options.cutoff, &d_cell_check};

    // Launch box check kernel
    box_check_kernel->launch(
        dim3(1),        // grid size
        dim3(32),       // block size
        0,              // shared memory
        nullptr,        // stream
        box_check_args, // arguments
        true            // synchronize
    );

    // Check box validity, assume fail
    int h_cell_check = 1;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int), cudaMemcpyDeviceToHost));

    if (debug_timing_enabled()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::high_resolution_clock::now() - _box_check_start
        )
                           .count();
        fprintf(stderr, "[TIMING] box_check (NVRTC): %ld us\n", elapsed);
        fflush(stderr);
    }

    if (h_cell_check != 0) {
        throw std::runtime_error("Invalid cutoff: too large for box dimensions");
    }

    // Decide whether to use cell list or brute force
    // Auto defaults to cell list since it's faster for most practical system sizes
    bool use_cell_list;
    switch (options.algorithm) {
    case VesinBruteForce:
        use_cell_list = false;
        break;
    case VesinCellList:
    case VesinAutoAlgorithm:
    default:
        use_cell_list = true;
        break;
    }

    if (use_cell_list) {
        // =====================================================================
        // Cell List Path
        // =====================================================================
        NVTX_PUSH("cell_list_total");

        // Ensure cell list buffers are allocated with fixed MAX_CELLS capacity
        // This avoids D2H sync to read n_cells_total
        NVTX_PUSH("ensure_buffers");
        ensure_cell_list_buffers(extras->cell_list, n_points, MAX_CELLS);
        NVTX_POP();
        auto& cl = extras->cell_list;

        int max_cells_int = static_cast<int>(MAX_CELLS);
        int min_particles_per_cell = MIN_PARTICLES_PER_CELL;

        // NVRTC kernels (compiled at runtime)
        const int THREADS_PER_BLOCK = 256;
        int num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Kernel 0: Compute cell grid parameters on device
        NVTX_PUSH("kernel0_grid_params");
        auto* grid_kernel = factory.create(
            "compute_cell_grid_params",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> grid_args = {
            &d_box, &d_periodic, &options.cutoff, &max_cells_int, &n_points, &min_particles_per_cell, &cl.inv_box, &cl.n_cells, &cl.n_search, &cl.n_cells_total
        };
        grid_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_args, false);
        NVTX_POP();

        // Zero cell counts for MAX_CELLS (fixed size, no D2H sync needed)
        NVTX_PUSH("memset_cell_counts");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cl.cell_counts, 0, sizeof(int) * MAX_CELLS));
        NVTX_POP();

        // Kernel 1: Assign cell indices
        NVTX_PUSH("kernel1_assign_cells");
        auto* assign_kernel = factory.create(
            "assign_cell_indices",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> assign_args = {
            &d_positions, &cl.inv_box, &d_periodic, &cl.n_cells, &n_points, &cl.cell_indices, &cl.particle_shifts
        };
        assign_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, assign_args, false
        );
        NVTX_POP();

        // Kernel 2: Count particles per cell
        NVTX_PUSH("kernel2_count_particles");
        auto* count_kernel = factory.create(
            "count_particles_per_cell",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> count_args = {
            &cl.cell_indices, &n_points, &cl.cell_counts
        };
        count_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, count_args, false
        );
        NVTX_POP();

        // Kernel 3: Parallel prefix sum (reads n_cells_total from device)
        NVTX_PUSH("kernel3_prefix_sum");
        auto* prefix_kernel = factory.create(
            "prefix_sum_cells",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> prefix_args = {
            &cl.cell_counts, &cl.cell_starts, &cl.n_cells_total
        };
        // Use 256 threads, shared memory for thread chunk totals
        int prefix_threads = 256;
        size_t shared_mem = sizeof(int) * prefix_threads;
        prefix_kernel->launch(
            dim3(1), dim3(prefix_threads), shared_mem, nullptr, prefix_args, false
        );
        NVTX_POP();

        // Copy cell_starts to cell_offsets (working copy for scatter)
        NVTX_PUSH("memcpy_cell_offsets");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            cl.cell_offsets, cl.cell_starts, sizeof(int) * MAX_CELLS, cudaMemcpyDeviceToDevice
        ));
        NVTX_POP();

        // Kernel 4: Scatter particles
        NVTX_PUSH("kernel4_scatter");
        auto* scatter_kernel = factory.create(
            "scatter_particles",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> scatter_args = {
            &d_positions, &cl.cell_indices, &cl.particle_shifts, &cl.cell_offsets, &n_points, &cl.sorted_positions, &cl.sorted_indices, &cl.sorted_shifts, &cl.sorted_cell_indices
        };
        scatter_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, scatter_args, false
        );
        NVTX_POP();

        // Kernel 5: Find neighbors using optimized multi-thread-per-particle approach
        NVTX_PUSH("kernel5_find_neighbors");
        auto* find_kernel = factory.create(
            "find_neighbors_optimized",
            CELL_LIST_CODE,
            "cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> find_args = {
            &cl.sorted_positions, &cl.sorted_indices, &cl.sorted_shifts, &cl.sorted_cell_indices, &cl.cell_starts, &cl.cell_counts, &d_box, &d_periodic, &cl.n_cells, &cl.n_search, &n_points, &options.cutoff, &options.full, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
        };
        // With THREADS_PER_PARTICLE=8, each warp handles 4 particles
        // Each block of 256 threads handles 32 particles
        const int THREADS_PER_PARTICLE = 8;
        int particles_per_block = THREADS_PER_BLOCK / THREADS_PER_PARTICLE;
        int num_blocks_find = (n_points + particles_per_block - 1) / particles_per_block;
        find_kernel->launch(
            dim3(num_blocks_find), dim3(THREADS_PER_BLOCK), 0, nullptr, find_args, false
        );
        NVTX_POP();

        NVTX_POP(); // cell_list_total
    }

    if (!use_cell_list) {
        // =====================================================================
        // Brute Force Path
        // =====================================================================
        NVTX_PUSH("brute_force_total");

        // Prepare arguments for neighbor computation kernel
        std::vector<void*> args = {
            &d_positions, &d_box, &d_periodic, &n_points, &options.cutoff, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
        };

        // Launch appropriate neighbor computation kernel
        const int WARP_SIZE = 32;
        const int NWARPS = 4;
        dim3 blockDim(WARP_SIZE * NWARPS);

        if (options.full) {
            // Full neighbor list kernel
            NVTX_PUSH("brute_force_full_kernel");
            auto* full_kernel = factory.create(
                "compute_mic_neighbours_full_impl",
                CUDA_CODE,
                "mic_neighbourlist.cu",
                {"-std=c++17", "-DNWARPS=" + std::to_string(NWARPS), "-DWARP_SIZE=" + std::to_string(WARP_SIZE)}
            );

            dim3 gridDim(std::max((int)(n_points + NWARPS - 1) / NWARPS, 1));

            full_kernel->launch(gridDim, blockDim, 0, nullptr, args, false);
            NVTX_POP();

        } else {
            // Half neighbor list kernel
            NVTX_PUSH("brute_force_half_kernel");
            auto* half_kernel = factory.create(
                "compute_mic_neighbours_half_impl",
                CUDA_CODE,
                "mic_neighbourlist.cu",
                {"-std=c++17", "-DNWARPS=" + std::to_string(NWARPS), "-DWARP_SIZE=" + std::to_string(WARP_SIZE)}
            );

            const size_t num_all_pairs = n_points * (n_points - 1) / 2;
            int threads_per_block = WARP_SIZE * NWARPS;
            int num_blocks = (num_all_pairs + threads_per_block - 1) / threads_per_block;
            dim3 gridDim(std::max(num_blocks, 1));

            half_kernel->launch(gridDim, blockDim, 0, nullptr, args, false);
            NVTX_POP();
        }

        NVTX_POP(); // brute_force_total
    }

    // Copy final pair count from device to pinned host memory using async copy
    // Then synchronize and read from pinned memory
    NVTX_PUSH("async_copy_and_sync");

    // Start timing for final sync
    auto _final_sync_start = std::chrono::high_resolution_clock::now();
    if (debug_timing_enabled()) {
        // Don't sync here - we want to measure the actual async kernel completion time
        _final_sync_start = std::chrono::high_resolution_clock::now();
    }

    // Async copy to pinned memory (faster than sync cudaMemcpy to regular memory)
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr // default stream
    ));

    // Synchronize to ensure the copy is complete
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize());

    if (debug_timing_enabled()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - _final_sync_start).count();
        fprintf(stderr, "[TIMING] final_sync: %ld us\n", elapsed);
        fflush(stderr);
    }

    // Read from pinned memory
    neighbors.length = *extras->pinned_length_ptr;

    if (debug_timing_enabled()) {
        auto total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::high_resolution_clock::now() - _function_start
        )
                                 .count();
        fprintf(stderr, "[TIMING] total_c++ function: %ld us\n", total_elapsed);
        fflush(stderr);
    }

    NVTX_POP();
}
