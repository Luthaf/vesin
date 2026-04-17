#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <optional>
#include <stdexcept>

#define NOMINMAX
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

const unsigned char CUDA_BRUTEFORCE_CODE[] = {
#include "generated/cuda_bruteforce.cu.inc"
};

const unsigned char CUDA_CELL_LIST_CODE[] = {
#include "generated/cuda_cell_list.cu.inc"
};

// Maximum number of cells (limited by single-block prefix sum)
static constexpr size_t DEFAULT_MAX_CELLS = 8192;
// Minimum particles per cell target for good GPU utilization.
// Lower values create more cells and reduce per-cell neighbor work, which is
// beneficial on larger systems where more coarse grids become too dense.
static constexpr size_t MIN_PARTICLES_PER_CELL = 8;

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
    size_t n_periodic = 0;
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
    if (box_diag_out != nullptr) {
        box_diag_out[0] = a_norm;
        box_diag_out[1] = b_norm;
        box_diag_out[2] = c_norm;
    }

    // Compute and output inverse box (needed for general PBC)
    if ((inv_box_out != nullptr) && !is_orthogonal) {
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
        double bc_cross[3];
        double ac_cross[3];
        double ab_cross[3];
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
    GPULITE_CUDART_CALL(cudaFree(cl.cell_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.particle_shifts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_counts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_starts));
    GPULITE_CUDART_CALL(cudaFree(cl.cell_offsets));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_positions));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_shifts));
    GPULITE_CUDART_CALL(cudaFree(cl.sorted_cell_indices));
    GPULITE_CUDART_CALL(cudaFree(cl.inv_box));
    GPULITE_CUDART_CALL(cudaFree(cl.n_cells));
    GPULITE_CUDART_CALL(cudaFree(cl.n_search));
    GPULITE_CUDART_CALL(cudaFree(cl.n_cells_total));
    GPULITE_CUDART_CALL(cudaFree(cl.bounding_min));
    GPULITE_CUDART_CALL(cudaFree(cl.bounding_max));

    cl = CellListBuffers();
}

CudaNeighborListExtras::~CudaNeighborListExtras() {
    if (this->length_ptr != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->length_ptr);
    }
    if (this->cell_check_ptr != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->cell_check_ptr);
    }
    if (this->overflow_flag != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->overflow_flag);
    }
    if (this->box_diag != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->box_diag);
    }
    if (this->inv_box_brute != nullptr) {
        gpulite::CUDART::instance().cudaFree(this->inv_box_brute);
    }
    try {
        free_cell_list_buffers(this->cell_list);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error freeing cell list buffers: " << e.what() << std::endl;
    }
}

vesin::cuda::CudaNeighborListExtras*
vesin::cuda::get_cuda_extras(VesinNeighborList* neighbors) {
    if (neighbors->opaque == nullptr) {
        neighbors->opaque = new vesin::cuda::CudaNeighborListExtras();
        auto* test = static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
    }
    return static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors->opaque);
}

static void reset(VesinNeighborList& neighbors) {
    auto* extras = vesin::cuda::get_cuda_extras(&neighbors);

    if ((neighbors.pairs != nullptr) && is_device_ptr(getPtrAttributes(neighbors.pairs), "pairs")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.pairs));
    }
    if ((neighbors.shifts != nullptr) && is_device_ptr(getPtrAttributes(neighbors.shifts), "shifts")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.shifts));
    }
    if ((neighbors.distances != nullptr) && is_device_ptr(getPtrAttributes(neighbors.distances), "distances")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.distances));
    }
    if ((neighbors.vectors != nullptr) && is_device_ptr(getPtrAttributes(neighbors.vectors), "vectors")) {
        GPULITE_CUDART_CALL(cudaFree(neighbors.vectors));
    }

    neighbors.pairs = nullptr;
    neighbors.shifts = nullptr;
    neighbors.distances = nullptr;
    neighbors.vectors = nullptr;
    extras->length_ptr = nullptr;

    // Free pinned memory if allocated
    if (extras->pinned_length_ptr != nullptr) {
        GPULITE_CUDART_CALL(cudaFreeHost(extras->pinned_length_ptr));
        extras->pinned_length_ptr = nullptr;
    }

    // Free brute force buffers
    GPULITE_CUDART_CALL(cudaFree(extras->box_diag));
    extras->box_diag = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras->inv_box_brute));
    extras->inv_box_brute = nullptr;

    GPULITE_CUDART_CALL(cudaFree(extras->overflow_flag));
    extras->overflow_flag = nullptr;

    free_cell_list_buffers(extras->cell_list);

    *extras = CudaNeighborListExtras();
}

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {
    assert(neighbors.device.type == VesinCUDA);

    int32_t curr_device = -1;
    int32_t device_id = -1;

    if (neighbors.pairs != nullptr) {
        GPULITE_CUDART_CALL(cudaGetDevice(&curr_device));
        device_id = get_device_id(neighbors.pairs);

        if ((device_id != -1) && curr_device != device_id) {
            GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        }
    }

    reset(neighbors);

    if ((device_id != -1) && curr_device != device_id) {
        GPULITE_CUDART_CALL(cudaSetDevice(curr_device));
    }

    delete static_cast<vesin::cuda::CudaNeighborListExtras*>(neighbors.opaque);
    neighbors.opaque = nullptr;
}

void checkCuda() {
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

// Ensure cell list buffers are allocated with sufficient capacity
static void realloc_buffers_if_needed(
    CellListBuffers& cl,
    size_t n_points,
    size_t n_cells
) {
    if (cl.max_points < n_points) {
        // Free old point-related buffers
        GPULITE_CUDART_CALL(cudaFree(cl.cell_indices));
        GPULITE_CUDART_CALL(cudaFree(cl.particle_shifts));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_positions));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_indices));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_shifts));
        GPULITE_CUDART_CALL(cudaFree(cl.sorted_cell_indices));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.particle_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_positions, sizeof(double) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_indices, sizeof(int32_t) * n_points));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_shifts, sizeof(int32_t) * n_points * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.sorted_cell_indices, sizeof(int32_t) * n_points));
        cl.max_points = n_points;
    }

    if (cl.max_cells < n_cells) {
        // Free old cell-related buffers
        GPULITE_CUDART_CALL(cudaFree(cl.cell_counts));
        GPULITE_CUDART_CALL(cudaFree(cl.cell_starts));
        GPULITE_CUDART_CALL(cudaFree(cl.cell_offsets));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_counts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_starts, sizeof(int32_t) * n_cells));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.cell_offsets, sizeof(int32_t) * n_cells));
        cl.max_cells = n_cells;
    }

    // Allocate cell grid parameter buffers (fixed size, only once)
    if (cl.inv_box == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.inv_box, sizeof(double) * 9));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_cells, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_search, sizeof(int32_t) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.n_cells_total, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.bounding_min, sizeof(double) * 3));
        GPULITE_CUDART_CALL(cudaMalloc((void**)&cl.bounding_max, sizeof(double) * 3));
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
    if (options.sorted) {
        throw std::runtime_error("CUDA implemented does not support sorted output yet");
    }

    // Check if CUDA is available
    checkCuda();

    // check that all pointers are are device pointers
    if (!is_device_ptr(getPtrAttributes(points), "points")) {
        throw std::runtime_error("`points` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(getPtrAttributes(box), "box")) {
        throw std::runtime_error("`box` pointer is not allocated on a CUDA device");
    }

    if (!is_device_ptr(getPtrAttributes(periodic), "periodic")) {
        throw std::runtime_error("`periodic` pointer is not allocated on a CUDA device");
    }

    auto device_id = get_device_id(points);
    if (device_id != get_device_id(box)) {
        throw std::runtime_error("`points` and `box` do not exist on the same device");
    }

    if (device_id != get_device_id(periodic)) {
        throw std::runtime_error("`points` and `periodic` do not exist on the same device");
    }

    if (device_id != neighbors.device.device_id) {
        throw std::runtime_error("`points`, `box` and `periodic` device differs from input neighbors device_id");
    }

    auto* extras = vesin::cuda::get_cuda_extras(&neighbors);
    size_t max_pairs_per_point = std::max(
        static_cast<size_t>(VESIN_CUDA_AT_LEAST_PAIRS_PER_POINT),
        static_cast<size_t>(std::ceil(std::pow(options.cutoff, 3)))
    );

    if (extras->allocated_device_id != device_id) {
        // first switch to previous device
        if (extras->allocated_device_id >= 0) {
            GPULITE_CUDART_CALL(cudaSetDevice(extras->allocated_device_id));
        }
        // free any existing allocations
        reset(neighbors);
        // switch back to current device
        GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        extras->allocated_device_id = device_id;
    }

    if (extras->capacity >= n_points && (extras->length_ptr != nullptr)) {
        GPULITE_CUDART_CALL(cudaMemset(extras->length_ptr, 0, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));
    } else {
        auto saved_device = extras->allocated_device_id;
        reset(neighbors);
        extras->allocated_device_id = saved_device;

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

        extras->max_pairs = n_points * max_pairs_per_point;

        GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.pairs, sizeof(size_t) * extras->max_pairs * 2));

        if (options.return_shifts) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.shifts, sizeof(int32_t) * extras->max_pairs * 3));
        }

        if (options.return_distances) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.distances, sizeof(double) * extras->max_pairs));
        }

        if (options.return_vectors) {
            GPULITE_CUDART_CALL(cudaMalloc((void**)&neighbors.vectors, sizeof(double) * extras->max_pairs * 3));
        }

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->length_ptr, sizeof(size_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->length_ptr, 0, sizeof(size_t)));

        // Pinned host memory for async D2H copy
        GPULITE_CUDART_CALL(cudaHostAlloc(
            (void**)&extras->pinned_length_ptr,
            sizeof(size_t),
            cudaHostAllocDefault
        ));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->cell_check_ptr, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->cell_check_ptr, 0, sizeof(int32_t)));

        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->overflow_flag, sizeof(int32_t)));
        GPULITE_CUDART_CALL(cudaMemset(extras->overflow_flag, 0, sizeof(int32_t)));

        extras->capacity = static_cast<size_t>(1.2 * static_cast<double>(n_points));
    }

    const auto* d_positions = reinterpret_cast<const double*>(points);
    const auto* d_box = reinterpret_cast<const double*>(box);
    const auto* d_periodic = periodic;

    auto* d_pair_indices = reinterpret_cast<size_t*>(neighbors.pairs);
    auto* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    auto* d_distances = neighbors.distances;
    auto* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    auto* d_pair_counter = extras->length_ptr;
    auto* d_cell_check = extras->cell_check_ptr;
    auto* d_overflow_flag = extras->overflow_flag;
    size_t max_pairs = extras->max_pairs;

    auto& factory = gpulite::KernelFactory::instance(device_id);

    auto cuda_bruteforce_code = std::string(reinterpret_cast<const char*>(CUDA_BRUTEFORCE_CODE), sizeof(CUDA_BRUTEFORCE_CODE));
    auto cuda_cell_list_code = std::string(reinterpret_cast<const char*>(CUDA_CELL_LIST_CODE), sizeof(CUDA_CELL_LIST_CODE));

    if (extras->box_diag == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->box_diag, sizeof(double) * 3));
    }
    if (extras->inv_box_brute == nullptr) {
        GPULITE_CUDART_CALL(cudaMalloc((void**)&extras->inv_box_brute, sizeof(double) * 9));
    }

    auto* box_check_kernel = factory.create(
        "mic_box_check",
        cuda_bruteforce_code,
        "cuda_bruteforce.cu",
        {"-std=c++17", "-default-device"}
    );

    double* d_box_diag = extras->box_diag;
    double* d_inv_box_brute = extras->inv_box_brute;
    std::vector<void*> box_check_args = {
        static_cast<void*>(&d_box),
        static_cast<void*>(&d_periodic),
        static_cast<void*>(&options.cutoff),
        static_cast<void*>(&d_cell_check),
        static_cast<void*>(&d_box_diag),
        static_cast<void*>(&d_inv_box_brute),
    };

    box_check_kernel->launch(dim3(1), dim3(32), 0, nullptr, box_check_args, false);

    int32_t h_cell_check = 1;
    GPULITE_CUDART_CALL(cudaMemcpy(&h_cell_check, d_cell_check, sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool box_check_error = (h_cell_check & 1) != 0;
    bool is_orthogonal = (h_cell_check & 2) != 0;

    // Get box dimensions for auto algorithm selection
    double h_box_diag[3];
    GPULITE_CUDART_CALL(cudaMemcpy(h_box_diag, d_box_diag, sizeof(double) * 3, cudaMemcpyDeviceToHost));
    double min_box_dim = std::min({h_box_diag[0], h_box_diag[1], h_box_diag[2]});
    bool cutoff_requires_cell_list = options.cutoff > min_box_dim / 2.0;

    bool use_cell_list;
    switch (options.algorithm) {
    case VesinBruteForce:
        if (box_check_error) {
            throw std::runtime_error("Invalid cutoff: too large for box dimensions");
        }
        use_cell_list = false;
        break;
    case VesinCellList:
        use_cell_list = true;
        break;
    case VesinAutoAlgorithm:
    default:
        // Use cell list if cutoff > half box size, or for large/non-orthogonal systems
        use_cell_list = cutoff_requires_cell_list || !is_orthogonal || n_points >= 5000;
        break;
    }

    if (use_cell_list) {
        NVTX_PUSH("cell_list_total");

        // Compute effective max cells based on minimum particles per cell target
        // This ensures we have enough work per cell for good GPU utilization
        size_t max_cells_from_particles = std::max(1UL, n_points / MIN_PARTICLES_PER_CELL);
        size_t max_cells = std::min(DEFAULT_MAX_CELLS, max_cells_from_particles);

        NVTX_PUSH("ensure_buffers");
        realloc_buffers_if_needed(extras->cell_list, n_points, max_cells);
        NVTX_POP();
        auto& cl = extras->cell_list;

        size_t THREADS_PER_BLOCK = 256;
        size_t num_blocks_points = (n_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        NVTX_PUSH("kernel0_bounding_box");
        auto* bounding_kernel = factory.create(
            "compute_bounding_box",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> bounding_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        // the 256 here must match the size of shared memory allocated inside the code,
        // if you update one please update the other.
        bounding_kernel->launch(dim3(1), dim3(256), 0, nullptr, bounding_args, false);
        NVTX_POP();

        NVTX_PUSH("kernel1_grid_params");
        auto* grid_kernel = factory.create(
            "compute_cell_grid_params",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> grid_args = {
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&max_cells),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&cl.n_search),
            static_cast<void*>(&cl.n_cells_total),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        grid_kernel->launch(dim3(1), dim3(1), 0, nullptr, grid_args, false);
        NVTX_POP();

        NVTX_PUSH("memset_cell_counts_starts");
        GPULITE_CUDART_CALL(cudaMemset(cl.cell_counts, 0, sizeof(int32_t) * max_cells));
        GPULITE_CUDART_CALL(cudaMemset(cl.cell_starts, 0, sizeof(int32_t) * max_cells));
        NVTX_POP();

        NVTX_PUSH("kernel1_assign_cells");
        auto* assign_kernel = factory.create(
            "assign_cell_indices",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> assign_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cl.inv_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&cl.particle_shifts),
            static_cast<void*>(&cl.bounding_min),
            static_cast<void*>(&cl.bounding_max),
        };
        assign_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, assign_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel2_count_particles");
        auto* count_kernel = factory.create(
            "count_particles_per_cell",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> count_args = {
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.cell_counts),
        };
        count_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, count_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel3_prefix_sum");
        auto* prefix_kernel = factory.create(
            "prefix_sum_cells",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> prefix_args = {
            static_cast<void*>(&cl.cell_counts),
            static_cast<void*>(&cl.cell_starts),
            static_cast<void*>(&cl.n_cells_total),
        };
        size_t prefix_threads = 256;
        size_t shared_mem = sizeof(int32_t) * prefix_threads;
        prefix_kernel->launch(
            dim3(1), dim3(prefix_threads), shared_mem, nullptr, prefix_args, false
        );
        NVTX_POP();

        NVTX_PUSH("memcpy_cell_offsets");
        GPULITE_CUDART_CALL(cudaMemcpy(
            cl.cell_offsets, cl.cell_starts, sizeof(int32_t) * max_cells, cudaMemcpyDeviceToDevice
        ));
        NVTX_POP();

        NVTX_PUSH("kernel4_scatter");
        auto* scatter_kernel = factory.create(
            "scatter_particles",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> scatter_args = {
            static_cast<void*>(&d_positions),
            static_cast<void*>(&cl.cell_indices),
            static_cast<void*>(&cl.particle_shifts),
            static_cast<void*>(&cl.cell_offsets),
            static_cast<void*>(&n_points),
            static_cast<void*>(&cl.sorted_positions),
            static_cast<void*>(&cl.sorted_indices),
            static_cast<void*>(&cl.sorted_shifts),
            static_cast<void*>(&cl.sorted_cell_indices),
        };
        scatter_kernel->launch(
            dim3(num_blocks_points), dim3(THREADS_PER_BLOCK), 0, nullptr, scatter_args, false
        );
        NVTX_POP();

        NVTX_PUSH("kernel5_find_neighbors");
        auto* find_kernel = factory.create(
            "find_neighbors_optimized",
            cuda_cell_list_code,
            "cuda_cell_list.cu",
            {"-std=c++17", "-default-device"}
        );
        std::vector<void*> find_args = {
            static_cast<void*>(&cl.sorted_positions),
            static_cast<void*>(&cl.sorted_indices),
            static_cast<void*>(&cl.sorted_shifts),
            static_cast<void*>(&cl.sorted_cell_indices),
            static_cast<void*>(&cl.cell_starts),
            static_cast<void*>(&cl.cell_counts),
            static_cast<void*>(&d_box),
            static_cast<void*>(&d_periodic),
            static_cast<void*>(&cl.n_cells),
            static_cast<void*>(&cl.n_search),
            static_cast<void*>(&n_points),
            static_cast<void*>(&options.cutoff),
            static_cast<void*>(&options.full),
            static_cast<void*>(&d_pair_counter),
            static_cast<void*>(&d_pair_indices),
            static_cast<void*>(&d_shifts),
            static_cast<void*>(&d_distances),
            static_cast<void*>(&d_vectors),
            static_cast<void*>(&options.return_shifts),
            static_cast<void*>(&options.return_distances),
            static_cast<void*>(&options.return_vectors),
            static_cast<void*>(&max_pairs),
            static_cast<void*>(&d_overflow_flag)
        };
        size_t THREADS_PER_PARTICLE = 8;
        size_t particles_per_block = THREADS_PER_BLOCK / THREADS_PER_PARTICLE;
        size_t num_blocks_find = (n_points + particles_per_block - 1) / particles_per_block;
        find_kernel->launch(
            dim3(num_blocks_find), dim3(THREADS_PER_BLOCK), 0, nullptr, find_args, false
        );
        NVTX_POP();

        NVTX_POP(); // cell_list_total
    }

    if (!use_cell_list) {
        NVTX_PUSH("brute_force_total");

        size_t THREADS_PER_BLOCK = 128;
        double cutoff2 = options.cutoff * options.cutoff;

        size_t num_half_pairs = n_points * (n_points - 1) / 2;

        if (is_orthogonal) {
            if (options.full) {
                NVTX_PUSH("brute_force_full_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_full_orthogonal",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box_diag),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_orthogonal");
                auto* kernel = factory.create(
                    "brute_force_half_orthogonal",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box_diag),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            }
        } else {
            if (options.full) {
                NVTX_PUSH("brute_force_full_general");
                auto* kernel = factory.create(
                    "brute_force_full_general",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box),
                    static_cast<void*>(&d_inv_box_brute),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            } else {
                NVTX_PUSH("brute_force_half_general");
                auto* kernel = factory.create(
                    "brute_force_half_general",
                    cuda_bruteforce_code,
                    "cuda_bruteforce.cu",
                    {"-std=c++17", "-default-device"}
                );

                std::vector<void*> args = {
                    static_cast<void*>(&d_positions),
                    static_cast<void*>(&d_box),
                    static_cast<void*>(&d_inv_box_brute),
                    static_cast<void*>(&d_periodic),
                    static_cast<void*>(&n_points),
                    static_cast<void*>(&cutoff2),
                    static_cast<void*>(&d_pair_counter),
                    static_cast<void*>(&d_pair_indices),
                    static_cast<void*>(&d_shifts),
                    static_cast<void*>(&d_distances),
                    static_cast<void*>(&d_vectors),
                    static_cast<void*>(&options.return_shifts),
                    static_cast<void*>(&options.return_distances),
                    static_cast<void*>(&options.return_vectors),
                    static_cast<void*>(&max_pairs),
                    static_cast<void*>(&d_overflow_flag)
                };

                size_t num_blocks = (num_half_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                kernel->launch(
                    /*grid=*/dim3(std::max(num_blocks, static_cast<size_t>(1))),
                    /*block=*/dim3(THREADS_PER_BLOCK),
                    /*shared_mem_size=*/0,
                    /*cuda_stream=*/nullptr,
                    /*args=*/args,
                    /*synchronize=*/false
                );
                NVTX_POP();
            }
        }

        NVTX_POP(); // brute_force_total
    }

    NVTX_PUSH("async_copy_and_sync");

    GPULITE_CUDART_CALL(cudaMemcpyAsync(
        extras->pinned_length_ptr,
        d_pair_counter,
        sizeof(size_t),
        cudaMemcpyDeviceToHost,
        nullptr
    ));

    GPULITE_CUDART_CALL(cudaDeviceSynchronize());

    // Check for overflow
    int h_overflow_flag = 0;
    GPULITE_CUDART_CALL(cudaMemcpy(
        &h_overflow_flag,
        d_overflow_flag,
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    if (h_overflow_flag != 0) {
        throw std::runtime_error(
            "The number of neighbor pairs exceeds the maximum capacity of " +
            std::to_string(max_pairs) + " (max_pairs_per_point=" +
            std::to_string(max_pairs_per_point) + "; n_points=" +
            std::to_string(n_points) + "). " +
            "Consider reducing the cutoff distance, or explicitly setting " +
            "VESIN_CUDA_MAX_PAIRS_PER_POINT as an environment variable."
        );
    }

    neighbors.length = *extras->pinned_length_ptr;

    NVTX_POP();
}
