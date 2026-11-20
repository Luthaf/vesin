#include <cmath>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include <gpulite/gpulite.hpp>

#include "./neighbors.hpp"
#include "./pick_algo.hpp"
#include "./utils.hpp"

#include "kernels/bruteforce.cuh"

// This file picks which of the two CUDA neighbor search algorithms
// `VesinAutoAlgorithm` should run. The cell list wins above a certain number of
// points and the brute force below it, and where exactly that happens is
// measured on the device in use the first time it is asked.
//
// Where the cell list starts to beat the brute force on one device is stored as
// a number of points per unit of reduced cutoff: the crossover is assumed to
// grow linearly with the reduced cutoff, i.e. the cutoff divided by the typical
// distance between points `(volume / n_points)^(1/3)`. This makes it
// independent of the unit of length used for the points and cutoff.
//
// `scripts/algorithms-crossover.py` measures this to update the default, and
// must be updated if this model changes.

using namespace vesin::cuda;

// What to use when the device could not be measured. Use
// `scripts/algorithms-crossover.py` to re-measure it.
static constexpr double FALLBACK_CROSSOVER = 500.0;

// Hard lower and upper bounds on the crossover, to avoid picking a cell list
// for a couple hundred points or a brute force for tens of thousands.
static constexpr double MIN_CROSSOVER_POINTS = 200.0;
static constexpr double MAX_CROSSOVER_POINTS = 50000.0;

// A simple cubic lattice filling `counts[dim] * spacing` in each direction,
// together with the box holding it.
static std::vector<double> calibration_lattice(const int32_t counts[3], double box[3][3], double spacing) {
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            box[i][j] = 0.0;
        }
        box[i][i] = counts[i] * spacing;
    }

    auto points = std::vector<double>();
    points.reserve(3 * static_cast<size_t>(counts[0]) * counts[1] * counts[2]);
    for (int32_t x = 0; x < counts[0]; x++) {
        for (int32_t y = 0; y < counts[1]; y++) {
            for (int32_t z = 0; z < counts[2]; z++) {
                points.push_back(x * spacing);
                points.push_back(y * spacing);
                points.push_back(z * spacing);
            }
        }
    }
    return points;
}

namespace {

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
    int32_t device_id,
    const std::vector<double>& points,
    const double box[3][3],
    VesinAlgorithm algorithm,
    double cutoff,
    int32_t n_iters
) {
    constexpr int32_t N_WARMUP = 1;

    auto n_points = points.size() / 3;
    bool periodic[3] = {true, true, true};

    auto device_guard = vesin::cuda::DeviceGuard(device_id);

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
        options.sorted = false;
        options.algorithm = algorithm;
        options.return_shifts = false;
        options.return_distances = false;
        options.return_vectors = false;
        options.skin = 0.0;

        for (int32_t i = 0; i < N_WARMUP + n_iters; i++) {
            if (i == N_WARMUP) {
                GPULITE_CUDART_CALL(cudaStreamSynchronize(nullptr));
                timings.clear();
            }
            auto start = std::chrono::steady_clock::now();
            // The calibration only touches its own buffers, and runs on the
            // legacy default stream regardless of the caller's stream.
            vesin::cuda::neighbors(
                d_points, n_points, d_box, d_periodic, /*stream=*/nullptr, options, neighbors
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
// Both algorithms are timed on lattices of increasing size, until the cell list
// becomes faster than the brute force. The crossing is then interpolated in
// log-log space between the last two sizes. Interpolating over a wider range of
// sizes is not accurate, since `log(cell list / brute force)` is far from linear
// in `log(n_points)`: on large GPUs, both algorithms take a constant time for
// small systems, before the brute force starts growing as `n_points^2`.
static std::optional<double> measure_crossover(int32_t device_id) {
    constexpr double CUTOFF = 5.0;
    constexpr double SPACING = 2.0;

    // Number of points along each side of the lattices, from 216 to 32768
    // points. The crossover is usually between 500 and 5000 points for this
    // cutoff, and we stop as soon as the cell list is faster.
    static constexpr int32_t SIDES[] = {6, 8, 10, 12, 14, 16, 20, 24, 28, 32};

    // Stop the sweep once a single brute force run takes more than this (in
    // microseconds), to bound the time spent here when the brute force stays
    // faster for a long time (e.g. on a slow GPU).
    constexpr double MAX_BRUTE_FORCE_TIME = 5000.0;

    auto previous_log_n = 0.0;
    auto previous_log_ratio = 0.0;
    for (auto side : SIDES) {
        auto log_n = 3.0 * std::log(static_cast<double>(side));

        int32_t counts[3] = {side, side, side};
        double box[3][3] = {};
        auto points = calibration_lattice(counts, box, SPACING);

        // the brute force gets expensive for the largest systems
        auto n_iters = side <= 20 ? 9 : 3;
        auto brute = time_neighbors(device_id, points, box, VesinBruteForce, CUTOFF, n_iters);
        auto cell = time_neighbors(device_id, points, box, VesinCellList, CUTOFF, n_iters);
        if (!(brute > 0.0) || !(cell > 0.0)) {
            return std::nullopt;
        }

        auto log_ratio = std::log(cell / brute);
        if (log_ratio < 0.0) {
            if (side == SIDES[0]) {
                // the cell list is already faster for the smallest system, use
                // its size as the crossover
                return std::exp(log_n) / (CUTOFF / SPACING);
            }

            auto fraction = previous_log_ratio / (previous_log_ratio - log_ratio);
            auto crossover = std::exp(previous_log_n + fraction * (log_n - previous_log_n));
            return crossover / (CUTOFF / SPACING);
        }

        if (brute > MAX_BRUTE_FORCE_TIME) {
            // the brute force is still faster, but already slow enough that
            // the cell list is a good choice above this size
            return std::exp(log_n) / (CUTOFF / SPACING);
        }

        previous_log_n = log_n;
        previous_log_ratio = log_ratio;
    }

    // the brute force is still faster for the largest system, use its size as
    // the crossover
    return std::exp(previous_log_n) / (CUTOFF / SPACING);
}

// Where the two algorithms cost the same on one device, measured the first time
// that device picks an algorithm and kept for the rest of the process. Empty if
// they could not be measured at all.
static const std::optional<double>& crossover(int32_t device_id) {
    static std::mutex mutex;
    static std::map<int32_t, std::optional<double>> measured;

    auto guard = std::lock_guard<std::mutex>(mutex);
    auto it = measured.find(device_id);
    if (it != measured.end()) {
        return it->second;
    }

    std::optional<double> measurement = std::nullopt;
    try {
        measurement = measure_crossover(device_id);
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
    double points_volume,
    int32_t device_id
) {
    // Past the brute force kernels' own limits the cell list is the only one
    // that can run at all.
    if (n_points > VESIN_MAX_BRUTE_FORCE_POINTS || n_images > VESIN_MAX_BRUTE_FORCE_IMAGES) {
        return true;
    }

    auto spacing = std::cbrt(points_volume / n_points);
    auto n_crossover = crossover(device_id).value_or(FALLBACK_CROSSOVER) * cutoff / spacing;

    // `fmin`/`fmax` ignore NaN, so a degenerate volume ends up on the lower
    // bound, and picks the cell list, which can handle any system.
    return n_points >= std::fmin(std::fmax(n_crossover, MIN_CROSSOVER_POINTS), MAX_CROSSOVER_POINTS);
}
