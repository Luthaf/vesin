#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#include <algorithm>
#include <optional>
#include <stdexcept>

#include <gpulite/gpulite.hpp>

#include "pick_cuda_algo.hpp"
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

#include "cuda/scan.cuh"
const unsigned char CUDA_SCAN_HEX[] = {
#include "generated/cuda/scan.cu.inc"
};
const char* CUDA_SCAN_CODE = reinterpret_cast<const char*>(CUDA_SCAN_HEX);

// Hard upper bound on the number of cells, to keep the per-cell arrays (three
// `int32_t` each) at a sane size. This only kicks in for very large systems:
// the per-system budget below is normally the binding constraint.
static constexpr size_t DEFAULT_MAX_CELLS = 1 << 21;

// Upper bound on the number of cells per particle. This budget only matters for
// dilute systems, where the cell size picked by `compute_cell_grid_params`
// would create far more cells than particles and most of them would be empty.
// Empty cells are pure overhead: they cost a cell visit in the neighbor search
// without contributing any candidate pair.
static constexpr size_t MAX_CELLS_PER_PARTICLE = 2;

// Block size for the two neighbor search kernels, and the number of threads
// they put on each atom. One atom's search is a walk over (2 * n_search + 1)^3
// cells, and splitting it across threads both multiplies the number of blocks
// and shortens the dependent chain of gather loads within one atom.
//
// Both must match `CELL_LIST_BLOCK` and `THREADS_PER_PARTICLE` in
// cuda/cell_list.cu, which uses them to size its shared memory. The block size
// must be a multiple of 32 and the thread count must divide 32, so that a group
// never straddles two warps.
static constexpr size_t CELL_LIST_BLOCK = 256;
static constexpr size_t THREADS_PER_PARTICLE = 8;

// Block size for the two bounding box kernels. Must be a power of two for their
// tree reduction, and must match `BOUNDING_BOX_BLOCK` in cuda/cell_list.cu,
// which uses it to size its shared memory.
static constexpr size_t BOUNDING_BOX_BLOCK = 256;

// Block size for `compute_cell_grid_params`. Must match `GRID_PARAMS_BLOCK` in
// cuda/cell_list.cu.
static constexpr size_t GRID_PARAMS_BLOCK = 256;

// Cost of visiting one neighbor cell, relative to the cost of evaluating one
// candidate pair. `compute_cell_grid_params` uses this to pick the cell size:
// the smaller it is, the finer the grid we can afford.
//
// Fitted from timings of the diamond benchmark: a least-squares fit of
//
//     time = A * total_candidates + B * total_cell_visits + C
//
// gives B/A = 1.23. A cell visit is therefore slightly *more* expensive than a
// candidate evaluation.
static constexpr double CELL_VISIT_COST = 1.23;

/// Compile (or fetch from cache) an NVRTC kernel. Each macro is bound to one of
/// the embedded CUDA sources, so call sites only need the factory and the
/// kernel function name:
///
///   auto* kernel = GET_CELL_LIST_KERNEL(factory, count_neighbors_cell_list);
#define GET_CELL_LIST_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_CELL_LIST_CODE, "cell_list.cu", {"-std=c++17", "-default-device"})

#define GET_BRUTEFORCE_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_BRUTEFORCE_CODE, "bruteforce.cu", {"-std=c++17", "-default-device"})

#define GET_SORT_PAIRS_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_SORT_PAIRS_CODE, "sort_pairs.cu", {"-std=c++17", "-default-device"})

#define GET_SCAN_KERNEL(factory, name) \
    factory.create<decltype(name)>(#name, CUDA_SCAN_CODE, "scan.cu", {"-std=c++17", "-default-device"})

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

// Allocate the small, fixed-size scratch buffers held by `extras`. These are
// independent of the pair capacity and are only ever allocated once.
static void allocate_scratch_buffers(CudaNeighborListExtras* extras) {
    if (extras->d_length_ptr == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_length_ptr, sizeof(size_t)));
    }

    if (extras->pinned == nullptr) {
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned,
            sizeof(PinnedReadback),
            cudaHostAllocDefault
        ));
    }

    if (extras->d_all_periodic == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_all_periodic, sizeof(bool)));
    }

    if (extras->d_overflow_flag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_overflow_flag, sizeof(int32_t)));
    }
}

void vesin::cuda::allocate_output_buffers(
    VesinNeighborList& neighbors,
    size_t n_pairs,
    VesinOptions options
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);
    allocate_scratch_buffers(extras);

    bool missing_requested_output =
        (options.return_shifts && neighbors.shifts == nullptr) ||
        (options.return_distances && neighbors.distances == nullptr) ||
        (options.return_vectors && neighbors.vectors == nullptr);

    if (extras->pairs_capacity >= n_pairs && !missing_requested_output) {
        return;
    }

    free_output_buffers(neighbors);

    size_t capacity = std::max(extras->pairs_capacity, n_pairs);
    extras->pairs_capacity = 0;

    GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * capacity * 2));

    if (options.return_shifts) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * capacity * 3));
    }

    if (options.return_distances) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * capacity));
    }

    if (options.return_vectors) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * capacity * 3));
    }

    extras->pairs_capacity = capacity;
}

struct BoxChecks {
    bool use_cell_list = false;
    bool all_periodic = false;
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

    if (extras->d_n_search == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->d_n_search, sizeof(int32_t) * 3));
    }
    if (extras->d_inv_box == nullptr) {
        double (*ptr)[3] = nullptr;
        GPULITE_CUDART_CALL(cudaMalloc((void**)&ptr, sizeof(double[3][3])));
        extras->d_inv_box = ptr;
    }

    auto* box_check_kernel = GET_BRUTEFORCE_KERNEL(factory, analyze_box);

    // `analyze_box` is a single-thread kernel, there are only 18 numbers to
    // look at and everything downstream needs the answer before it can start
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);

    box_check_kernel->launch(
        config,
        d_box,
        d_periodic,
        options.cutoff,
        extras->d_all_periodic,
        extras->d_n_search,
        extras->d_inv_box
    );

    auto* pinned = extras->pinned;
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->all_periodic, extras->d_all_periodic,
        sizeof(bool), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        pinned->n_search, extras->d_n_search,
        sizeof(int32_t) * 3, cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    results.all_periodic = pinned->all_periodic;

    // How many periodic images of each pair the brute force kernels would walk.
    // This is 1 for a box at least twice the cutoff in every periodic
    // direction, and grows as the box gets smaller than that.
    const int32_t* n_search = pinned->n_search;
    size_t n_images = 1;
    for (size_t dim = 0; dim < 3; dim++) {
        n_images *= 2 * static_cast<size_t>(n_search[dim]) + 1;
    }

    switch (options.algorithm) {
    case VesinBruteForce: {
        if (n_points > MAX_BRUTE_FORCE_POINTS) {
            throw std::runtime_error(
                "too many points (" + std::to_string(n_points) + ") for the "
                "brute force algorithm (the limit is " + std::to_string(MAX_BRUTE_FORCE_POINTS) + "); "
                "use the cell list instead"
            );
        }

        if (n_images > MAX_BRUTE_FORCE_IMAGES) {
            throw std::runtime_error(
                "the cutoff (" + std::to_string(options.cutoff) + ") is so large "
                "compared to the box that the brute force algorithm would have "
                "to check " + std::to_string(n_images) + " periodic images of "
                "every pair of points (the limit is " +
                std::to_string(MAX_BRUTE_FORCE_IMAGES) + "); use the cell list "
                "instead"
            );
        }

        results.use_cell_list = false;
        break;
    }
    case VesinCellList:
        results.use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        results.use_cell_list = auto_use_cell_list(
            static_cast<double>(n_points),
            static_cast<double>(n_images),
            options.cutoff,
            options.sorted,
            neighbors.device.device_id
        );
        break;
    }

    return results;
}

// Device-side exclusive prefix sum of `d_in[n]` → `d_out[n]`.
//
// `d_block_totals` and `d_block_offsets` must each hold at least `ceil(n/1024)`
// elements, and `d_block_totals2` at least `ceil(n/1024^2)`.
//
// This scans the whole `max_cells` range rather than just the `n_cells_total`
// cells actually in use: the count of an unused cell is zero (`d_cell_counts`
// is zeroed above) so it contributes nothing, and this keeps the number of
// elements host-side, avoiding a read back of `n_cells_total`.
static void device_exclusive_scan(
    gpulite::KernelFactory& factory,
    const int32_t* d_in,
    int32_t* d_out,
    int32_t* d_block_totals,
    int32_t* d_block_offsets,
    int32_t* d_block_totals2,
    size_t n
) {
    // These must match the `SCAN_*` defines in cuda/scan.cu.
    size_t scan_threads = 256;
    size_t scan_elements_per_block = 1024;
    size_t scan_max_blocks = 256;

    size_t n_scan_blocks = (n + scan_elements_per_block - 1) / scan_elements_per_block;
    size_t n_scan_blocks_2 = (n_scan_blocks + scan_elements_per_block - 1) / scan_elements_per_block;

    if (n_scan_blocks_2 > scan_max_blocks) {
        throw std::runtime_error(
            "too many elements (" + std::to_string(n) + ") for the two-level "
            "device prefix scan (the limit is ~68 billion). This should not happen."
        );
    }

    auto* scan_phase_1 = GET_SCAN_KERNEL(factory, exclusive_scan_phase_1);
    auto* scan_phase_2 = GET_SCAN_KERNEL(factory, exclusive_scan_phase_2);
    auto* scan_phase_3 = GET_SCAN_KERNEL(factory, exclusive_scan_phase_3);

    auto config = gpulite::LaunchConfig();
    config.blockDim = dim3(scan_threads);

    // Each block of 256 threads processes 1024 elements (4 per thread):
    //   - Each thread loads and sums its 4 elements
    //   - Hillis-Steele inclusive scan over the 256 per-thread sums (shared mem)
    //   - Each thread writes the exclusive scan of its 4 elements to d_out
    //   - The last thread stores the block's total to d_block_totals[blockIdx.x]
    //
    // After this, d_out holds the correct exclusive scan *within* each block,
    // but blocks don't know their offset relative to earlier blocks.
    // d_block_totals[i] = total count of elements in blocks 0..i.
    config.gridDim = dim3(n_scan_blocks);
    scan_phase_1->launch(config, d_in, d_out, d_block_totals, n);

    if (n_scan_blocks == 1) {
        // A single block covers the whole input, so the local scan phase 1 just
        // wrote is already the global one: phase 2 would only set the single
        // block total to zero, and phase 3 would add that zero back. Skipping
        // both saves two kernel launches, which is worth having for the small
        // systems where the launch latency is the whole cost of the call.
        return;
    }

    if (n_scan_blocks <= scan_max_blocks) {
        config.gridDim = dim3(1);
        scan_phase_2->launch(config, d_block_totals, n_scan_blocks);

        config.gridDim = dim3(n_scan_blocks);
        scan_phase_3->launch(config, d_out, d_block_totals, n);
    } else {
        // The level-1 block totals array has more than 256 elements, so it
        // can't be scanned by a single phase-2 block. We apply the same 3-phase
        // scan to d_block_totals itself, producing per-block offsets that are
        // then added back at both levels.
        //
        // Level 2 phase 1: scan d_block_totals → d_block_offsets (local scan
        //   within each level-2 block), collect level-2 block totals in
        //   d_block_totals2. d_block_offsets now holds correct values within
        //   each level-2 block, but not globally.
        //
        // Level 2 phase 2: exclusive scan of d_block_totals2 in a single block
        //   (≤256 elements by construction). After this, d_block_totals2[i] =
        //   the offset to add to level-2 block i's results.
        //
        // Level 2 phase 3: add level-2 offsets to d_block_offsets. After this,
        //   d_block_offsets holds the correct global exclusive prefix sum of
        //   the level-1 block totals — i.e. d_block_offsets[i] = the global
        //   offset for level-1 block i.
        //
        // Level 1 phase 3: add d_block_offsets[blockIdx.x] to every element in
        //   d_out. After this, d_out holds the correct global exclusive prefix
        //   sum of d_in.

        // Level 2 phase 1
        config.gridDim = dim3(n_scan_blocks_2);
        scan_phase_1->launch(config, d_block_totals, d_block_offsets, d_block_totals2, n_scan_blocks);

        // Level 2 phase 2
        config.gridDim = dim3(1);
        scan_phase_2->launch(config, d_block_totals2, n_scan_blocks_2);

        // Level 2 phase 3: add level-2 offsets to level-1 block offsets
        config.gridDim = dim3(n_scan_blocks_2);
        scan_phase_3->launch(config, d_block_offsets, d_block_totals2, n_scan_blocks);

        // Level 1 phase 3: add level-1 block offsets to per-element scan
        config.gridDim = dim3(n_scan_blocks);
        scan_phase_3->launch(config, d_out, d_block_offsets, n);
    }
}

// Exclusive prefix sum of the per-atom neighbour counts
// `d_counts[n_points]` → `d_write_offsets[n_points]`, returning the total
// number of pairs.
static size_t device_exclusive_prefix_sum(
    gpulite::KernelFactory& factory,
    CellListBuffersCUDA& cl,
    PinnedReadback* pinned,
    size_t n_points
) {
    device_exclusive_scan(
        factory,
        cl.d_counts,
        cl.d_write_offsets,
        cl.d_block_totals,
        cl.d_block_offsets,
        cl.d_block_totals2,
        n_points
    );

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->last_offset, &cl.d_write_offsets[n_points - 1],
        sizeof(int32_t), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &pinned->last_count, &cl.d_counts[n_points - 1],
        sizeof(int32_t), cudaMemcpyDeviceToHost, nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    int32_t h_last_offset = pinned->last_offset;
    int32_t h_last_count = pinned->last_count;

    auto total = static_cast<int64_t>(h_last_offset) + h_last_count;
    if (h_last_offset < 0 || h_last_count < 0 || total > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "too many pairs for the CUDA cell list (the limit is 2^31); "
            "please open an issue on GitHub if you need more than that"
        );
    }

    return static_cast<size_t>(total);
}

// Reorder the output pair list (and any optional aligned outputs) so that pairs
// come out sorted by the first particle index. Used by the bruteforce and
// Verlet paths — the cell-list path produces output already sorted.
void vesin::cuda::sort_pairs(
    size_t n_points,
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    if (neighbors.length > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "too many pairs to sort on the GPU (the limit is 2^31); "
            "please open an issue on GitHub if you need more than that"
        );
    }

    NVTX_PUSH("sort_pairs");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    auto& sort = extras->sort_buffers;
    sort.allocate(
        neighbors.length,
        n_points,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    auto* copy_and_count_kernel = GET_SORT_PAIRS_KERNEL(factory, sort_pairs_copy_and_count);
    auto* scatter_kernel = GET_SORT_PAIRS_KERNEL(factory, sort_pairs_scatter);

    const size_t sort_threads = 256;
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3((neighbors.length + sort_threads - 1) / sort_threads);
    config.blockDim = dim3(sort_threads);

    // The pass below accumulates into `d_counts` with atomics, so it has to
    // start from zero on every call.
    GPULITE_CUDART_CALL(cudaMemset(sort.d_counts, 0, sizeof(int32_t) * n_points));

    copy_and_count_kernel->launch(
        config,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        sort.d_pairs_tmp,
        sort.d_shifts_tmp,
        sort.d_distances_tmp,
        sort.d_vectors_tmp,
        sort.d_counts,
        neighbors.length,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    device_exclusive_scan(
        factory,
        sort.d_counts,
        sort.d_offsets,
        sort.d_block_totals,
        sort.d_block_offsets,
        sort.d_block_totals2,
        n_points
    );

    scatter_kernel->launch(
        config,
        sort.d_pairs_tmp,
        sort.d_shifts_tmp,
        sort.d_distances_tmp,
        sort.d_vectors_tmp,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors,
        sort.d_offsets,
        neighbors.length,
        options.return_shifts,
        options.return_distances,
        options.return_vectors
    );

    NVTX_POP();
}

static void run_cell_list(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    bool all_periodic,
    VesinNeighborList& neighbors
) {
    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("cell_list_total");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    size_t max_cells = std::max(n_points * MAX_CELLS_PER_PARTICLE, static_cast<size_t>(1));
    max_cells = std::min(DEFAULT_MAX_CELLS, max_cells);

    NVTX_PUSH("ensure_buffers");
    extras->cell_list.allocate(n_points, max_cells);
    NVTX_POP();
    auto& cl = extras->cell_list;

    auto config = gpulite::LaunchConfig();

    // The bounding box is only ever read for non-periodic directions, skip the
    // work if everything is periodic
    if (!all_periodic) {
        NVTX_PUSH("kernel0_bounding_box");
        auto* bounding_kernel = GET_CELL_LIST_KERNEL(factory, compute_bounding_box);

        size_t bbox_blocks = std::min(
            std::max((n_points + BOUNDING_BOX_BLOCK - 1) / BOUNDING_BOX_BLOCK, static_cast<size_t>(1)),
            MAX_BOUNDING_BOX_BLOCKS
        );

        config.gridDim = dim3(bbox_blocks);
        config.blockDim = dim3(BOUNDING_BOX_BLOCK);
        bounding_kernel->launch(
            config,
            d_points,
            n_points,
            cl.d_bbox_partials,
            cl.d_face_distances,
            cl.d_bounding_min
        );

        // With a single block there is nothing to combine and the kernel above
        // already wrote the final answer.
        if (bbox_blocks > 1) {
            auto* reduce_kernel = GET_CELL_LIST_KERNEL(factory, reduce_bounding_box);

            config.gridDim = dim3(1);
            reduce_kernel->launch(
                config,
                cl.d_bbox_partials,
                bbox_blocks,
                cl.d_face_distances,
                cl.d_bounding_min
            );
        }
        NVTX_POP();
    }

    NVTX_PUSH("kernel1_grid_params");
    auto* grid_kernel = GET_CELL_LIST_KERNEL(factory, compute_cell_grid_params);

    config.gridDim = dim3(1);
    config.blockDim = dim3(GRID_PARAMS_BLOCK);
    grid_kernel->launch(
        config,
        d_box,
        d_periodic,
        options.cutoff,
        n_points,
        max_cells,
        CELL_VISIT_COST,
        extras->d_inv_box,
        cl.d_n_cells,
        cl.d_n_search,
        cl.d_n_cells_total,
        cl.d_face_distances,
        cl.d_cell_out_of_reach
    );
    NVTX_POP();

    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_counts, 0, sizeof(int32_t) * max_cells));
    GPULITE_CUDART_CALL(cudaMemset(cl.d_cell_starts, 0, sizeof(int32_t) * max_cells));

    NVTX_PUSH("kernel1_assign_cells");
    auto* assign_kernel = GET_CELL_LIST_KERNEL(factory, assign_cell_indices);

    size_t THREADS_PER_BLOCK = 256;
    size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

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
    auto* count_kernel = GET_CELL_LIST_KERNEL(factory, count_particles_per_cell);

    count_kernel->launch(
        config,
        cl.d_cell_indices,
        n_points,
        cl.d_cell_counts
    );
    NVTX_POP();

    // Exclusive prefix sum of the per-cell counts.
    NVTX_PUSH("kernel3_prefix_sum");
    device_exclusive_scan(
        factory,
        cl.d_cell_counts,
        cl.d_cell_starts,
        cl.d_cell_block_totals,
        cl.d_cell_block_offsets,
        cl.d_cell_block_totals2,
        max_cells
    );
    NVTX_POP();

    NVTX_PUSH("memcpy_cell_offsets");
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        cl.d_cell_offsets, cl.d_cell_starts,
        sizeof(int32_t) * max_cells, cudaMemcpyDeviceToDevice, nullptr
    ));
    NVTX_POP();

    NVTX_PUSH("kernel4_scatter");
    auto* scatter_kernel = GET_CELL_LIST_KERNEL(factory, scatter_particles);

    config.gridDim = dim3(num_blocks_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    scatter_kernel->launch(
        config,
        d_points,
        n_points,
        cl.d_cell_indices,
        cl.d_particle_shifts,
        d_box,
        cl.d_cell_offsets,
        cl.d_sorted_points,
        cl.d_wrapped_positions,
        cl.d_sorted_indices,
        cl.d_sorted_shifts,
        cl.d_sorted_cell_indices
    );
    NVTX_POP();

    NVTX_PUSH("kernel5_count_neighbors");
    auto* count_neighbors_kernel = GET_CELL_LIST_KERNEL(factory, count_neighbors_cell_list);

    size_t particles_per_block = CELL_LIST_BLOCK / THREADS_PER_PARTICLE;
    size_t num_blocks_search = (n_points + particles_per_block - 1) / particles_per_block;

    config.gridDim = dim3(num_blocks_search);
    config.blockDim = dim3(CELL_LIST_BLOCK);

    count_neighbors_kernel->launch(
        config,
        cl.d_wrapped_positions,
        n_points,
        d_box,
        cl.d_cell_out_of_reach,
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
        cl.d_counts
    );
    NVTX_POP();

    // Device-side exclusive prefix sum of d_counts → d_write_offsets.
    NVTX_PUSH("prefix_sum_counts");
    size_t n_pairs = device_exclusive_prefix_sum(factory, cl, extras->pinned, n_points);
    NVTX_POP();

    // Allocate the output buffers. We ask for a little more than the number of
    // pairs we actually found, so that small step-to-step fluctuations don't
    // re-allocate all the buffers on every call.
    auto extra_pairs = std::min(n_pairs / 4, static_cast<size_t>(4096));
    vesin::cuda::allocate_output_buffers(neighbors, n_pairs + extra_pairs, options);

    NVTX_PUSH("kernel6_fill_neighbors");
    auto* fill_neighbors_kernel = GET_CELL_LIST_KERNEL(factory, fill_neighbors_cell_list);

    config.gridDim = dim3(num_blocks_search);
    config.blockDim = dim3(CELL_LIST_BLOCK);

    fill_neighbors_kernel->launch(
        config,
        cl.d_wrapped_positions,
        n_points,
        d_box,
        cl.d_cell_out_of_reach,
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
        cl.d_write_offsets,
        neighbors.pairs,
        neighbors.shifts,
        neighbors.distances,
        neighbors.vectors
    );
    NVTX_POP();

    neighbors.length = n_pairs;

    NVTX_POP(); // cell_list_total
}

// Run the brute force kernels, leaving the pairs they found in `neighbors`.
//
// Returns false when the output buffers were too small to hold them all. The
// exact number of pairs the search found will be in `neighbors.length`.
static bool run_brute_force(
    const double (*d_points)[3],
    size_t n_points,
    const double d_box[3][3],
    const bool d_periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    assert(n_points <= MAX_BRUTE_FORCE_POINTS);

    auto* extras = static_cast<CudaNeighborListExtras*>(neighbors.opaque);

    NVTX_PUSH("brute_force_total");
    auto& factory = gpulite::KernelFactory::instance(neighbors.device.device_id);

    GPULITE_CUDART_CALL(cudaMemset(extras->d_length_ptr, 0, sizeof(size_t)));
    GPULITE_CUDART_CALL(cudaMemset(extras->d_overflow_flag, 0, sizeof(int32_t)));

    size_t THREADS_PER_BLOCK = 128;
    auto config = gpulite::LaunchConfig();
    config.gridDim = dim3((n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, n_points);
    config.blockDim = dim3(THREADS_PER_BLOCK);

    if (options.full) {
        NVTX_PUSH("brute_force_full");
        auto* kernel = GET_BRUTEFORCE_KERNEL(factory, brute_force_full);

        kernel->launch(
            config,
            d_points,
            d_box,
            extras->d_inv_box,
            d_periodic,
            extras->d_n_search,
            options.cutoff * options.cutoff,
            options.return_shifts,
            options.return_distances,
            options.return_vectors,
            extras->d_length_ptr,
            neighbors.pairs,
            neighbors.shifts,
            neighbors.distances,
            neighbors.vectors,
            extras->pairs_capacity,
            extras->d_overflow_flag
        );
        NVTX_POP();
    } else {
        NVTX_PUSH("brute_force_half");
        auto* kernel = GET_BRUTEFORCE_KERNEL(factory, brute_force_half);

        kernel->launch(
            config,
            d_points,
            d_box,
            extras->d_inv_box,
            d_periodic,
            extras->d_n_search,
            options.cutoff * options.cutoff,
            options.return_shifts,
            options.return_distances,
            options.return_vectors,
            extras->d_length_ptr,
            neighbors.pairs,
            neighbors.shifts,
            neighbors.distances,
            neighbors.vectors,
            extras->pairs_capacity,
            extras->d_overflow_flag
        );
        NVTX_POP();
    }

    NVTX_POP(); // brute_force_total

    // Read back the pair count from the GPU-side atomic counter, and the
    // overflow flag.
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->length,
        extras->d_length_ptr,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        &extras->pinned->overflow,
        extras->d_overflow_flag,
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));
    GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));

    neighbors.length = extras->pinned->length;

    if (extras->pinned->overflow != 0) {
        return false;
    }

    // Brute-force output is unsorted; sort if requested
    if (options.sorted && neighbors.length > 1) {
        vesin::cuda::sort_pairs(n_points, options, neighbors);
    }

    return true;
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

    if (n_points == 0) {
        // Nothing to do, and the kernels below can not be launched with an
        // empty grid anyway.
        neighbors.length = 0;
        return;
    }

    if (n_points > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "Too many points (" + std::to_string(n_points) + ") for the CUDA "
            "neighbour list (limit is ~2.1 billion)."
        );
    }

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
    allocate_scratch_buffers(extras);

    if (options.skin > 0.0) {
        extras->verlet_cache.run(
            d_points, n_points, d_box, d_periodic, options, neighbors
        );
    } else {
        // no skin, free any remaining verlet buffers from a previous call
        extras->verlet_cache = VerletCache();

        auto box_checks = check_box(neighbors, d_box, d_periodic, n_points, options);
        if (box_checks.use_cell_list) {
            run_cell_list(
                d_points,
                n_points,
                d_box,
                d_periodic,
                options,
                box_checks.all_periodic,
                neighbors
            );
        } else {
            // Brute force writes its pairs through an atomic counter, and needs
            // a pre-allocated output. We guess a starting size here, and will
            // grow it below if needed
            size_t max_pairs_per_point = static_cast<size_t>(std::max(
                128.0, std::ceil(std::pow(options.cutoff, 3))
            ));

            vesin::cuda::allocate_output_buffers(
                neighbors, n_points * max_pairs_per_point, options
            );

            auto buffer_large_enough = run_brute_force(d_points, n_points, d_box, d_periodic, options, neighbors);

            if (!buffer_large_enough) {
                // The guess was too small, but the number of pairs is in
                // `neighbors.length`. We re-allocate the outputs with 10%
                // headroom in case the number of pairs fluctuate between now
                // and the next call.
                auto needed = static_cast<size_t>(1.1 * static_cast<double>(neighbors.length));
                vesin::cuda::allocate_output_buffers(neighbors, needed, options);

                buffer_large_enough = run_brute_force(d_points, n_points, d_box, d_periodic, options, neighbors);

                if (!buffer_large_enough) {
                    throw std::runtime_error(
                        "the brute force search overflowed its output buffers "
                        "twice in a row, the second time with room for " +
                        std::to_string(needed) + " pairs against the " +
                        std::to_string(neighbors.length) + " the first search "
                        "counted. This should not happen, please report it as a bug"
                    );
                }
            }
        }
    }
}
