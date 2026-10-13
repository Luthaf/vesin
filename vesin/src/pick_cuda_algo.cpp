#include <cmath>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include <gpulite/gpulite.hpp>

#include "cuda/bruteforce.cuh"

#include "pick_cuda_algo.hpp"
#include "vesin_cuda.hpp"

// This file picks which of the two CUDA neighbor search algorithms
// `VesinAutoAlgorithm` should run. The cell list wins above a certain number of
// points and the brute force below it, and where exactly that happens is
// measured on the device in use the first time it is asked.

using namespace vesin::cuda;

// Where the cell list starts to beat the brute force on one device.
//
// The cell list is already sorted, while brute force has to sort after the
// fact, so for sorted NL the crossover is smaller.
struct Crossovers {
    // Points per unit of cutoff, for unsorted output.
    double per_cutoff;
    // Points, with no cutoff term, for sorted output.
    double sorted;
};

// What to use when the device could not be measured.
//
// The benchmarks put every crossing within a factor of a few of these, for
// cutoffs from 2.5 to 12.
static constexpr Crossovers DEFAULT_CROSSOVERS = {500.0, 2500.0};

// Hard lower and upper bounds on the crossover, to avoid picking a cell list
// for a few hundred points or a brute force for tens of thousands.
static constexpr double MIN_CROSSOVER_POINTS = 500.0;
static constexpr double MAX_CROSSOVER_POINTS = 50000.0;

// A simple cubic lattice filling `counts[dim] * spacing` in each direction,
// together with the box holding it.
static std::vector<double> calibration_lattice(const int counts[3], double box[3][3], double spacing) {
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            box[i][j] = 0.0;
        }
        box[i][i] = counts[i] * spacing;
    }

    auto points = std::vector<double>();
    points.reserve(3 * static_cast<size_t>(counts[0]) * counts[1] * counts[2]);
    for (int x = 0; x < counts[0]; x++) {
        for (int y = 0; y < counts[1]; y++) {
            for (int z = 0; z < counts[2]; z++) {
                points.push_back(x * spacing);
                points.push_back(y * spacing);
                points.push_back(z * spacing);
            }
        }
    }
    return points;
}

namespace {

/// Guard that sets the current device to `device_id` on construction and
/// restores the previous device on destruction. This is used to ensure that the
/// calibration runs on the correct device, as the `neighbors` function allocates
/// memory on the current device and checks that the input pointers are on the
/// same device.
struct DeviceGuard {
    int previous = 0;
    explicit DeviceGuard(int device_id) {
        GPULITE_CUDART_CALL(cudaGetDevice(&this->previous));
        if (this->previous != device_id) {
            GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        } else {
            this->previous = -1;
        }
    }

    ~DeviceGuard() {
        if (this->previous >= 0) {
            gpulite::CUDART::instance().cudaSetDevice(this->previous);
        }
    }
};

/// RAII wrapper for a device buffer allocated with `cudaMalloc` and freed with
/// `cudaFree`.
struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        GPULITE_CUDART_CALL(cudaMalloc(&this->ptr, bytes));
    }

    ~DeviceBuffer() {
        if (this->ptr != nullptr) {
            gpulite::CUDART::instance().cudaFree(this->ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/// RAII wrapper for a `VesinNeighborList` that ensures its output buffers are
/// freed and its `opaque` pointer is deleted when it goes out of scope.
struct NeighborListRAII {
    VesinNeighborList list;

    ~NeighborListRAII() {
        if (this->list.opaque != nullptr) {
            vesin::cuda::free_neighbors(this->list);
        }
    }
};

}

// Run `n_iters` neighbor searches on the given points and return the median
// time one took, in microseconds. The first call is thrown away to allow it to
// compile the kernels and warm up the GPU.
static double time_neighbors(
    int device_id,
    const std::vector<double>& points,
    const double box[3][3],
    VesinAlgorithm algorithm,
    double cutoff,
    bool sorted,
    int n_iters
) {
    constexpr int N_WARMUP = 1;

    auto n_points = points.size() / 3;
    bool periodic[3] = {true, true, true};

    auto device_guard = DeviceGuard(device_id);

    // Allocate device buffers for the points, box, and periodicity flags.
    auto points_buffer = DeviceBuffer(sizeof(double) * points.size());
    auto box_buffer = DeviceBuffer(sizeof(double[3][3]));
    auto periodic_buffer = DeviceBuffer(sizeof(bool) * 3);

    auto* d_points = static_cast<double (*)[3]>(points_buffer.ptr);
    auto* d_box = static_cast<double (*)[3]>(box_buffer.ptr);
    auto* d_periodic = static_cast<bool*>(periodic_buffer.ptr);

    auto neighbors_guard = NeighborListRAII();
    auto& neighbors = neighbors_guard.list;
    neighbors.device = {VesinCUDA, device_id};

    auto timings = std::vector<double>();
    timings.reserve(static_cast<size_t>(n_iters));
    {
        GPULITE_CUDART_CALL(cudaMemcpy(
            d_points, points.data(), sizeof(double) * points.size(), cudaMemcpyHostToDevice
        ));
        GPULITE_CUDART_CALL(cudaMemcpy(
            d_box, box, sizeof(double[3][3]), cudaMemcpyHostToDevice
        ));
        GPULITE_CUDART_CALL(cudaMemcpy(
            d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice
        ));

        auto options = VesinOptions();
        options.cutoff = cutoff;
        options.full = false;
        options.sorted = sorted;
        options.algorithm = algorithm;
        options.return_shifts = false;
        options.return_distances = false;
        options.return_vectors = false;
        options.skin = 0.0;

        for (int i = 0; i < N_WARMUP + n_iters; i++) {
            if (i == N_WARMUP) {
                GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));
                timings.clear();
            }
            auto start = std::chrono::steady_clock::now();
            vesin::cuda::neighbors(
                d_points, n_points, d_box, d_periodic, options, neighbors
            );
            GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));
            auto stop = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
        }
    }

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

// Measure where the two algorithms cost the same on one device.
//
// `log(cell list / brute force)` falls through zero with a slope close to -1,
// brute force growing as `n^2` where the cell list grows as `n`, so two sizes
// on both sides of the crossing are measured and the crossing is interpolated.
static std::optional<Crossovers> measure_crossovers(int device_id) {
    constexpr double CUTOFF = 5.0;
    constexpr double SPACING = 2.0;
    #undef NAN
    constexpr double NAN = std::numeric_limits<double>::quiet_NaN();

    struct LogRatios {
        double plain;
        double sorted;
    };

    auto log_ratios = [&](int side, int n_iters) {
        int counts[3] = {side, side, side};
        double box[3][3] = {};
        auto points = calibration_lattice(counts, box, SPACING);

        auto brute = time_neighbors(
            device_id, points, box, VesinBruteForce, CUTOFF, /*sorted=*/false, n_iters
        );
        auto brute_sorted = time_neighbors(
            device_id, points, box, VesinBruteForce, CUTOFF, /*sorted=*/true, n_iters
        );
        // `neighbors` only sorts what the brute force and Verlet paths write,
        // the cell list already filling its pairs in order, so a single timing
        // covers the cell list both ways.
        auto cell = time_neighbors(
            device_id, points, box, VesinCellList, CUTOFF, /*sorted=*/false, n_iters
        );

        if (!(brute > 0.0) || !(brute_sorted > 0.0) || !(cell > 0.0)) {
            return LogRatios{NAN, NAN};
        }

        return LogRatios{std::log(cell / brute), std::log(cell / brute_sorted)};
    };

    static constexpr int SMALL_SIDE = 12;  // 1728 points
    static constexpr int LARGE_SIDE = 20;  // 8000 points

    auto small = log_ratios(SMALL_SIDE, /*n_iters=*/9);
    auto large = log_ratios(LARGE_SIDE, /*n_iters=*/3);

    auto log_small = std::log(std::pow(static_cast<double>(SMALL_SIDE), 3.0));
    auto log_large = std::log(std::pow(static_cast<double>(LARGE_SIDE), 3.0));

    // The cell list should start out behind and end up ahead. If this is not
    // the case, we fallback to the default crossover values.
    auto interpolate = [&](double at_small, double at_large) {
        if (std::isnan(at_small) || std::isnan(at_large)) {
            return NAN;
        }
        if (at_small <= 0.0 || at_large >= 0.0) {
            return NAN;
        }
        return std::exp(
            log_small + (log_large - log_small) * at_small / (at_small - at_large)
        );
    };

    auto plain = interpolate(small.plain, large.plain);
    auto sorted = interpolate(small.sorted, large.sorted);
    if (std::isnan(plain) || std::isnan(sorted)) {
        return std::nullopt;
    }

    return Crossovers{plain / CUTOFF, sorted};
}

// Where the two algorithms cost the same on one device, measured the first time
// that device picks an algorithm and kept for the rest of the process. Empty if
// they could not be measured at all.
static const std::optional<Crossovers>& crossovers(int device_id) {
    static std::mutex mutex;
    static std::map<int, std::optional<Crossovers>> measured;

    auto guard = std::lock_guard<std::mutex>(mutex);
    auto it = measured.find(device_id);
    if (it != measured.end()) {
        return it->second;
    }

    std::optional<Crossovers> measurement = std::nullopt;
    try {
        measurement = measure_crossovers(device_id);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "vesin: failed to measure crossover on device %d: %s\n", device_id, e.what()
        );
    }

    it = measured.emplace(device_id, measurement).first;
    return it->second;
}

// Should `VesinAutoAlgorithm` run the cell list?
bool vesin::cuda::auto_use_cell_list(
    double n_points,
    double n_images,
    double cutoff,
    bool sorted,
    int device_id
) {
    // Past the brute force kernels' own limits the cell list is the only one
    // that can run at all.
    if (n_points > MAX_BRUTE_FORCE_POINTS || n_images > MAX_BRUTE_FORCE_IMAGES) {
        return true;
    }

    auto measured = crossovers(device_id).value_or(DEFAULT_CROSSOVERS);
    auto crossover = sorted ? measured.sorted : measured.per_cutoff * cutoff;

    return n_points >= std::min(std::max(crossover, MIN_CROSSOVER_POINTS), MAX_CROSSOVER_POINTS);
}
