#include "vesin_cuda.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

using namespace vesin::cuda;

#ifndef VESIN_ENABLE_CUDA

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);
    // nothing to do, no data was allocated
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {
    throw std::runtime_error("vesin was not compiled with CUDA support");
}

CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    return nullptr;
}

#else

#include <optional>

#include <gpulite/gpulite.hpp>

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

static const char* CUDA_BRUTEFORCE_CODE =
#include "generated/cuda_bruteforce.cu"
    ;

static const char* CUDA_CELL_LIST_CODE =
#include "generated/cuda_cell_list.cu"
    ;

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t MAX_CELLS = 8192;
// Minimum particles per cell target for good GPU utilization
// Higher values = fewer cells = more work per block but larger search range
static constexpr int MIN_PARTICLES_PER_CELL = 128;

// Helper functions for CPU-side vector math
static inline double cpu_dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double cpu_norm3(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static inline void cpu_cross3(const double* a, const double* b, double* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void cpu_invert_matrix(const double* m, double* inv) {
    double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

    double inv_det = 1.0 / det;

    inv[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}

/// CPU-side box check that avoids GPU kernel launch overhead
/// Returns: {is_valid, is_orthogonal}
/// Also fills box_diag_out[3] and inv_box_out[9] if provided
static std::pair<bool, bool> cpu_box_check(
    const double h_box[9],
    const bool h_periodic[3],
    double cutoff,
    double* box_diag_out, // [3] output, can be nullptr
    double* inv_box_out   // [9] output, can be nullptr
) {
    const double* a = &h_box[0];
    const double* b = &h_box[3];
    const double* c = &h_box[6];

    double a_norm = cpu_norm3(a);
    double b_norm = cpu_norm3(b);
    double c_norm = cpu_norm3(c);

    // Count periodic directions
    int n_periodic = 0;
    if (h_periodic[0]) {
        n_periodic++;
    }
    if (h_periodic[1]) {
        n_periodic++;
    }
    if (h_periodic[2]) {
        n_periodic++;
    }

    double ab_dot = cpu_dot3(a, b);
    double ac_dot = cpu_dot3(a, c);
    double bc_dot = cpu_dot3(b, c);

    double tol = 1e-6;
    // Treat fully non-periodic systems as orthogonal
    // Also treat systems with zero-norm vectors as orthogonal (degenerate case)
    bool is_orthogonal = (n_periodic == 0) ||
                         (a_norm < tol || b_norm < tol || c_norm < tol) ||
                         ((std::fabs(ab_dot) < tol * a_norm * b_norm) &&
                          (std::fabs(ac_dot) < tol * a_norm * c_norm) &&
                          (std::fabs(bc_dot) < tol * b_norm * c_norm));

    // Output box diagonal (lengths)
    if (box_diag_out) {
        box_diag_out[0] = a_norm;
        box_diag_out[1] = b_norm;
        box_diag_out[2] = c_norm;
    }

    // Compute and output inverse box (needed for general PBC)
    if (inv_box_out && !is_orthogonal) {
        cpu_invert_matrix(h_box, inv_box_out);
    }

    // Compute minimum dimension for cutoff check
    double min_dim = 1e30;
    if (is_orthogonal) {
        if (h_periodic[0]) {
            min_dim = a_norm;
        }
        if (h_periodic[1]) {
            min_dim = std::fmin(min_dim, b_norm);
        }
        if (h_periodic[2]) {
            min_dim = std::fmin(min_dim, c_norm);
        }
    } else {
        // General case: compute perpendicular distances
        double bc_cross[3], ac_cross[3], ab_cross[3];
        cpu_cross3(b, c, bc_cross);
        cpu_cross3(a, c, ac_cross);
        cpu_cross3(a, b, ab_cross);

        double bc_norm = cpu_norm3(bc_cross);
        double ac_norm = cpu_norm3(ac_cross);
        double ab_norm = cpu_norm3(ab_cross);

        double V = std::fabs(cpu_dot3(a, bc_cross));

        double d_a = V / bc_norm;
        double d_b = V / ac_norm;
        double d_c = V / ab_norm;

        if (h_periodic[0]) {
            min_dim = d_a;
        }
        if (h_periodic[1]) {
            min_dim = std::fmin(min_dim, d_b);
        }
        if (h_periodic[2]) {
            min_dim = std::fmin(min_dim, d_c);
        }
    }

    bool is_valid = (cutoff * 2.0 <= min_dim);
    return {is_valid, is_orthogonal};
}

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
    if (this->box_diag) {
        CUDART_INSTANCE.cudaFree(this->box_diag);
    }
    if (this->inv_box_brute) {
        CUDART_INSTANCE.cudaFree(this->inv_box_brute);
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

    // Free brute force buffers
    if (extras->box_diag) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(extras->box_diag));
        extras->box_diag = nullptr;
    }
    if (extras->inv_box_brute) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaFree(extras->inv_box_brute));
        extras->inv_box_brute = nullptr;
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
    assert(neighbors.device.type == VesinCUDA);
    assert(!options.sorted && "Sorting is not supported in CUDA version of Vesin");

    // Check if CUDA is available
    checkCuda();

    // assert both points and box are device pointers
    assert(is_device_ptr(getPtrAttributes(points), "points") && "points pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(box), "box") && "box pointer is not allocated on a CUDA device");
    assert(is_device_ptr(getPtrAttributes(periodic), "periodic") && "periodic pointer is not allocated on a CUDA device");

    int device = get_device_id(points);
    // assert both points and box are on the same device
    assert((device == get_device_id(box)) && "`points` and `box` do not exist on the same device");
    assert((device == get_device_id(periodic)) && "`points` and `periodic` do not exist on the same device");
    assert((device == neighbors.device.device_id) && "`points`, `box` and `periodic` device differs from input neighbors device_id");

    auto extras = vesin::cuda::get_cuda_extras(&neighbors);

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

    if (extras->capacity >= n_points && extras->length_ptr) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->cell_check_ptr, 0, sizeof(int)));
    } else {
        int saved_device = extras->allocated_device;
        reset(neighbors);
        extras->allocated_device = saved_device;
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

        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        // Pinned host memory for async D2H copy
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

    auto& factory = KernelFactory::instance();

    if (!extras->box_diag) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->box_diag, sizeof(double) * 3));
    }
    if (!extras->inv_box_brute) {
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMalloc((void**)&extras->inv_box_brute, sizeof(double) * 9));
    }

    auto* box_check_kernel = factory.create(
        "mic_box_check",
        CUDA_BRUTEFORCE_CODE,
        "cuda_bruteforce.cu",
        {"-std=c++17"}
    );

    double* d_box_diag = extras->box_diag;
    double* d_inv_box_brute = extras->inv_box_brute;
    std::vector<void*> box_check_args = {&d_box, &d_periodic, &options.cutoff, &d_cell_check, &d_box_diag, &d_inv_box_brute};

    box_check_kernel->launch(dim3(1), dim3(32), 0, nullptr, box_check_args, false);

    int h_cell_check = 1;
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int), cudaMemcpyDeviceToHost));

    bool box_check_error = (h_cell_check & 1) != 0;
    bool is_orthogonal = (h_cell_check & 2) != 0;

    if (box_check_error) {
        throw std::runtime_error("Invalid cutoff: too large for box dimensions");
    }

    // Get box dimensions for auto algorithm selection
    double h_box_diag[3];
    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(h_box_diag, d_box_diag, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    double min_box_dim = std::min({h_box_diag[0], h_box_diag[1], h_box_diag[2]});
    bool cutoff_requires_cell_list = options.cutoff > min_box_dim / 2.0;

    bool use_cell_list;
    switch (options.algorithm) {
    case VesinBruteForce:
        use_cell_list = false;
        break;
    case VesinCellList:
        use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        // Use cell list if cutoff > half box size, or for large/non-orthogonal systems
        use_cell_list = cutoff_requires_cell_list || !(is_orthogonal && n_points < 5000);
        break;
    }

    if (use_cell_list) {
        NVTX_PUSH("cell_list_total");

        NVTX_PUSH("ensure_buffers");
        ensure_cell_list_buffers(extras->cell_list, n_points, MAX_CELLS);
        NVTX_POP();
        auto& cl = extras->cell_list;

        int max_cells_int = static_cast<int>(MAX_CELLS);
        int min_particles_per_cell = MIN_PARTICLES_PER_CELL;

        const int THREADS_PER_BLOCK = 256;
        int num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        NVTX_PUSH("kernel0_grid_params");
        auto* grid_kernel = factory.create(
            "compute_cell_grid_params",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> grid_args = {
            &d_box, &d_periodic, &options.cutoff, &max_cells_int, &n_points, &min_particles_per_cell, &cl.inv_box, &cl.n_cells, &cl.n_search, &cl.n_cells_total
        };
        grid_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_args, false);
        NVTX_POP();

        NVTX_PUSH("memset_cell_counts");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemset(cl.cell_counts, 0, sizeof(int) * MAX_CELLS));
        NVTX_POP();

        NVTX_PUSH("kernel1_assign_cells");
        auto* assign_kernel = factory.create(
            "assign_cell_indices",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> assign_args = {
            &d_positions, &cl.inv_box, &d_periodic, &cl.n_cells, &n_points, &cl.cell_indices, &cl.particle_shifts
        };
        assign_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, assign_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel2_count_particles");
        auto* count_kernel = factory.create(
            "count_particles_per_cell",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> count_args = {
            &cl.cell_indices, &n_points, &cl.cell_counts
        };
        count_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, count_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel3_prefix_sum");
        auto* prefix_kernel = factory.create(
            "prefix_sum_cells",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> prefix_args = {
            &cl.cell_counts, &cl.cell_starts, &cl.n_cells_total
        };
        int prefix_threads = 256;
        size_t shared_mem = sizeof(int) * prefix_threads;
        prefix_kernel->launch(
            dim3(1), dim3(prefix_threads), shared_mem, nullptr, prefix_args, false
        );
        NVTX_POP();

        NVTX_PUSH("memcpy_cell_offsets");
        CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpy(
            cl.cell_offsets, cl.cell_starts, sizeof(int) * MAX_CELLS, cudaMemcpyDeviceToDevice
        ));
        NVTX_POP();

        NVTX_PUSH("kernel4_scatter");
        auto* scatter_kernel = factory.create(
            "scatter_particles",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> scatter_args = {
            &d_positions, &cl.cell_indices, &cl.particle_shifts, &cl.cell_offsets, &n_points, &cl.sorted_positions, &cl.sorted_indices, &cl.sorted_shifts, &cl.sorted_cell_indices
        };
        scatter_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, scatter_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel5_find_neighbors");
        auto* find_kernel = factory.create(
            "find_neighbors_optimized",
            CUDA_CELL_LIST_CODE,
            "cuda_cell_list.cu",
            {"-std=c++17"}
        );
        std::vector<void*> find_args = {
            &cl.sorted_positions, &cl.sorted_indices, &cl.sorted_shifts, &cl.sorted_cell_indices, &cl.cell_starts, &cl.cell_counts, &d_box, &d_periodic, &cl.n_cells, &cl.n_search, &n_points, &options.cutoff, &options.full, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
        };
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
        NVTX_PUSH("brute_force_total");

        const int THREADS_PER_BLOCK = 128;
        double cutoff2 = options.cutoff * options.cutoff;

        size_t num_half_pairs = n_points * (n_points - 1) / 2;

        if (is_orthogonal) {
            if (options.full) {
                NVTX_PUSH("brute_force_full_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_full_orthogonal",
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
                );

                std::vector<void*> args = {
                    &d_positions, &d_box_diag, &d_periodic, &n_points, &cutoff2, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
                };

                int num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(dim3(std::max(num_blocks, 1)), dim3(THREADS_PER_BLOCK), 0, nullptr, args, false);
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_half_orthogonal",
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
                );

                std::vector<void*> args = {
                    &d_positions, &d_box_diag, &d_periodic, &n_points, &cutoff2, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
                };

                int num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(dim3(std::max(num_blocks, 1)), dim3(THREADS_PER_BLOCK), 0, nullptr, args, false);
                NVTX_POP();
            }
        } else {
            if (options.full) {
                NVTX_PUSH("brute_force_full_general");
                auto* kernel = factory.create(
                    "brute_force_full_general",
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
                );

                std::vector<void*> args = {
                    &d_positions, &d_box, &d_inv_box_brute, &d_periodic, &n_points, &cutoff2, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
                };

                int num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(dim3(std::max(num_blocks, 1)), dim3(THREADS_PER_BLOCK), 0, nullptr, args, false);
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_general");
                auto* kernel = factory.create(
                    "brute_force_half_general",
                    CUDA_BRUTEFORCE_CODE,
                    "cuda_bruteforce.cu",
                    {"-std=c++17"}
                );

                std::vector<void*> args = {
                    &d_positions, &d_box, &d_inv_box_brute, &d_periodic, &n_points, &cutoff2, &d_pair_counter, &d_pair_indices, &d_shifts, &d_distances, &d_vectors, &options.return_shifts, &options.return_distances, &options.return_vectors
                };

                int num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(dim3(std::max(num_blocks, 1)), dim3(THREADS_PER_BLOCK), 0, nullptr, args, false);
                NVTX_POP();
            }
        }

        NVTX_POP(); // brute_force_total
    }

    NVTX_PUSH("async_copy_and_sync");

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));

    CUDART_SAFE_CALL(CUDART_INSTANCE.cudaDeviceSynchronize());

    neighbors.length = *extras->pinned_length_ptr;

    NVTX_POP();
}

#endif
