#include <catch2/catch_test_macros.hpp>
#ifdef VESIN_TESTS_WITH_CUDA

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <vesin.h>

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        const char* message = cudaGetErrorString(status);
        FAIL(message);
    }
}

/// Neighbor list computed by `compute_on_device`, copied back to the host
struct DeviceResults {
    /// Empty if the calculation succeeded, the error message otherwise
    std::string error;
    std::vector<std::array<size_t, 2>> pairs;
    std::vector<std::array<int32_t, 3>> shifts;
    std::vector<double> distances;
    std::vector<std::array<double, 3>> vectors;
};

/// Compute a small neighbor list on the given device. This runs in multiple
/// threads at the same time, so it must not use Catch2 assertions (they are not
/// thread-safe): errors are reported in `DeviceResults::error` instead, and the
/// results are checked by `check_device_results` from the main thread.
static DeviceResults compute_on_device(int32_t device_id) {
    auto results = DeviceResults();
    // keep the first CUDA error
    auto check = [&](cudaError_t status) {
        if (status != cudaSuccess && results.error.empty()) {
            results.error = std::string("CUDA error: ") + cudaGetErrorString(status);
        }
    };

    double points[3][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    double box[3][3] = {
        {0.0, 3.0, 3.0},
        {3.0, 0.0, 3.0},
        {3.0, 3.0, 0.0},
    };
    bool periodic[3] = {true, true, true};

    check(cudaSetDevice(device_id));

    double (*d_points)[3] = nullptr;
    double (*d_box)[3] = nullptr;
    bool* d_periodic = nullptr;
    check(cudaMalloc(&d_points, sizeof(points)));
    check(cudaMalloc(&d_box, sizeof(box)));
    check(cudaMalloc(&d_periodic, sizeof(periodic)));
    check(cudaMemcpy(d_points, points, sizeof(points), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_box, box, sizeof(box), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_periodic, periodic, sizeof(periodic), cudaMemcpyHostToDevice));

    if (results.error.empty()) {
        auto options = VesinOptions();
        options.cutoff = 3.0;
        options.full = false;
        options.sorted = false;
        options.algorithm = VesinAutoAlgorithm;
        options.return_shifts = true;
        options.return_distances = true;
        options.return_vectors = true;

        auto neighbors = VesinNeighborList();
        const char* error_message = nullptr;
        auto status = vesin_neighbors(
            d_points, 3, d_box, d_periodic, {VesinCUDA, device_id}, options, &neighbors, &error_message
        );

        if (status != EXIT_SUCCESS) {
            results.error = std::string("vesin error: ") + (error_message != nullptr ? error_message : "unknown");
        } else {
            auto n_pairs = neighbors.length;
            results.pairs.resize(n_pairs);
            results.shifts.resize(n_pairs);
            results.distances.resize(n_pairs);
            results.vectors.resize(n_pairs);
            check(cudaMemcpy(results.pairs.data(), neighbors.pairs, sizeof(size_t[2]) * n_pairs, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(results.shifts.data(), neighbors.shifts, sizeof(int32_t[3]) * n_pairs, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(results.distances.data(), neighbors.distances, sizeof(double) * n_pairs, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(results.vectors.data(), neighbors.vectors, sizeof(double[3]) * n_pairs, cudaMemcpyDeviceToHost));
        }

        vesin_free(&neighbors);
    }

    check(cudaFree(d_points));
    check(cudaFree(d_box));
    check(cudaFree(d_periodic));

    return results;
}

static void check_device_results(const DeviceResults& results) {
    REQUIRE(results.error.empty());
    REQUIRE(results.pairs.size() == 5);

    for (size_t i = 0; i < results.pairs.size(); ++i) {
        // Which point of a half-list pair ends up as `i` and which as `j` is
        // not guaranteed: `(i, j)` with some shift and `(j, i)` with the
        // opposite shift describe the same pair. Canonicalize to `i < j` before
        // comparing, like `Pair::canonicalize()` in neighbors.cpp does.
        auto pair_i = results.pairs[i][0];
        auto pair_j = results.pairs[i][1];
        auto shift = results.shifts[i];
        auto vector = results.vectors[i];
        auto distance = results.distances[i];

        if (pair_i > pair_j) {
            std::swap(pair_i, pair_j);
            for (size_t d = 0; d < 3; d++) {
                shift[d] = -shift[d];
                vector[d] = -vector[d];
            }
        }

        if (pair_i == 0 && pair_j == 2) {
            // we have three pairs between 0 and 2 with shifts (-1, 0, 0),
            // (0, -1, 0), and (0, 0, -1)
            CHECK(distance == std::sqrt(6.0));

            if (shift[0] == -1 && shift[1] == 0 && shift[2] == 0) {
                CHECK(vector[0] == 2.0);
                CHECK(vector[1] == -1.0);
                CHECK(vector[2] == -1.0);
            } else if (shift[0] == 0 && shift[1] == -1 && shift[2] == 0) {
                CHECK(vector[0] == -1.0);
                CHECK(vector[1] == 2.0);
                CHECK(vector[2] == -1.0);
            } else if (shift[0] == 0 && shift[1] == 0 && shift[2] == -1) {
                CHECK(vector[0] == -1.0);
                CHECK(vector[1] == -1.0);
                CHECK(vector[2] == 2.0);
            } else {
                FAIL("Unexpected shift for pair (0, 2): (" + std::to_string(shift[0]) + ", " + std::to_string(shift[1]) + ", " + std::to_string(shift[2]) + ")");
            }

        } else if ((pair_i == 0 && pair_j == 1) || (pair_i == 1 && pair_j == 2)) {
            // pairs between 0-1 or 1-2 should have zero shifts, distance
            // sqrt(3), and vector (1, 1, 1)
            CHECK(shift[0] == 0);
            CHECK(shift[1] == 0);
            CHECK(shift[2] == 0);

            CHECK(distance == std::sqrt(3.0));
            CHECK(vector[0] == 1.0);
            CHECK(vector[1] == 1.0);
            CHECK(vector[2] == 1.0);
        } else {
            FAIL("Unexpected pair: (" + std::to_string(pair_i) + ", " + std::to_string(pair_j) + ")");
        }
    }
}

TEST_CASE("Test CUDA") {
    // get the number of CUDA devices
    int32_t n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);

    // start multiple threads at the same time on all devices, to test
    // concurrent execution. Each thread writes to its own entry in `results`.
    auto results = std::vector<DeviceResults>(10);
    auto threads = std::vector<std::thread>();
    for (size_t thread_id = 0; thread_id < results.size(); ++thread_id) {
        auto device_id = static_cast<int32_t>(thread_id) % n_devices;
        threads.emplace_back([&results, thread_id, device_id]() {
            results[thread_id] = compute_on_device(device_id);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (const auto& thread_results : results) {
        check_device_results(thread_results);
    }
}

// Exercise the CUDA Verlet skin cache path with a small system that
// reliably triggers the cache filter and displacement-check kernels.
TEST_CASE("Test CUDA Verlet skin cache") {
    int32_t n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);
    int32_t device_id = 0;
    check_cuda(cudaSetDevice(device_id));

    // small but sufficient system to exercise the Verlet cache kernels
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {4.0, 1.0, 0.0},
    };
    size_t n_points = 4;
    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double box[3][3] = {{10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}};
    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool periodic[3] = {true, true, true};
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    VesinNeighborList neighbors = {};

    auto options = VesinOptions();
    options.cutoff = 2.5;
    options.skin = 0.8; // enable Verlet cache path
    options.full = true;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        d_points, n_points, d_box, d_periodic, {VesinDeviceKind::VesinCUDA, device_id}, options, &neighbors, &error_message
    );
    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    // second call with small displacement to hit the cache filter kernels
    double moved_points[][3] = {
        {0.01, 0.01, 0.01},
        {1.02, 1.01, 0.99},
        {2.00, 2.03, 1.98},
        {4.01, 1.02, 0.01},
    };
    check_cuda(cudaMemcpy(d_points, moved_points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    status = vesin_neighbors(
        d_points, n_points, d_box, d_periodic, {VesinDeviceKind::VesinCUDA, device_id}, options, &neighbors, &error_message
    );
    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    vesin_free(&neighbors);
    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

// Copy the pairs of `neighbors` to the host, sorted to compare the output of
// algorithms producing them in an unspecified order
static std::vector<std::array<size_t, 2>> sorted_pairs(const VesinNeighborList& neighbors) {
    auto pairs = std::vector<std::array<size_t, 2>>(neighbors.length);
    check_cuda(cudaMemcpy(
        pairs.data(), neighbors.pairs, sizeof(size_t[2]) * neighbors.length, cudaMemcpyDeviceToHost
    ));
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

TEST_CASE("CUDA pre-allocated output arrays") {
    int32_t n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);
    check_cuda(cudaSetDevice(0));

    // simple cubic lattice of 4x4x4 points, each point has 18 neighbors with
    // a cutoff of 1.5
    auto points = std::vector<double>();
    for (size_t x = 0; x < 4; x++) {
        for (size_t y = 0; y < 4; y++) {
            for (size_t z = 0; z < 4; z++) {
                points.push_back(static_cast<double>(x));
                points.push_back(static_cast<double>(y));
                points.push_back(static_cast<double>(z));
            }
        }
    }
    size_t n_points = points.size() / 3;
    double box[3][3] = {{4.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 4.0}};
    bool periodic[3] = {true, true, true};

    double (*d_points)[3] = nullptr;
    double (*d_box)[3] = nullptr;
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * points.size()));
    check_cuda(cudaMemcpy(d_points, points.data(), sizeof(double) * points.size(), cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    int32_t status = EXIT_SUCCESS;
    auto compute = [&](VesinNeighborList& neighbors, VesinOptions options) -> std::string {
        const char* error_message = nullptr;
        status = vesin_neighbors(
            d_points, n_points, d_box, d_periodic, {VesinCUDA, 0}, options, &neighbors, &error_message
        );
        if (status != EXIT_SUCCESS) {
            return error_message;
        }
        return "";
    };

    // covers the brute force, the cell list, and the Verlet filter (with
    // skin > 0), with and without padding
    for (auto algorithm : {VesinBruteForce, VesinCellList}) {
        for (double skin : {0.0, 0.3}) {
            for (size_t padded_to : {0, 20}) {
                auto options = VesinOptions();
                options.cutoff = 1.5;
                options.full = true;
                options.algorithm = algorithm;
                options.skin = skin;
                options.padded_to = padded_to;
                options.return_shifts = true;
                options.return_distances = true;
                options.return_vectors = true;

                auto reference = VesinNeighborList();
                REQUIRE(compute(reference, options).empty());
                auto expected = sorted_pairs(reference);
                vesin_free(&reference);

                auto n_pairs = expected.size();
                REQUIRE(n_pairs == (padded_to > 0 ? n_points * padded_to : n_points * 18));

                size_t (*pairs)[2] = nullptr;
                int32_t (*shifts)[3] = nullptr;
                double* distances = nullptr;
                double (*vectors)[3] = nullptr;
                check_cuda(cudaMalloc(&pairs, sizeof(size_t[2]) * n_pairs));
                check_cuda(cudaMalloc(&shifts, sizeof(int32_t[3]) * n_pairs));
                check_cuda(cudaMalloc(&distances, sizeof(double) * n_pairs));
                check_cuda(cudaMalloc(&vectors, sizeof(double[3]) * n_pairs));

                auto neighbors = VesinNeighborList();
                auto set_arrays = [&](size_t capacity) {
                    neighbors.pairs = pairs;
                    neighbors.shifts = shifts;
                    neighbors.distances = distances;
                    neighbors.vectors = vectors;
                    neighbors.length = capacity;
                };

                options.pre_allocated_outputs = true;

                set_arrays(n_pairs);
                REQUIRE(compute(neighbors, options).empty());
                CHECK(neighbors.pairs == pairs);
                CHECK(sorted_pairs(neighbors) == expected);

                // the second call re-uses the Verlet cache if there is one
                set_arrays(n_pairs);
                REQUIRE(compute(neighbors, options).empty());
                CHECK(sorted_pairs(neighbors) == expected);

                set_arrays(n_pairs - 1);
                CHECK(compute(neighbors, options) == (
                    "the pre-allocated output arrays can hold " + std::to_string(n_pairs - 1) +
                    " pairs, but this calculation needs " + std::to_string(n_pairs) + " pairs"
                ));
                CHECK(status == VESIN_OUTPUT_TOO_SMALL);
                CHECK(neighbors.length == n_pairs);

                // with zero capacity, the arrays are not used (and can be
                // NULL), which allows counting pairs
                auto empty = VesinNeighborList();
                CHECK(!compute(empty, options).empty());
                CHECK(status == VESIN_OUTPUT_TOO_SMALL);
                CHECK(empty.length == n_pairs);
                vesin_free(&empty);

                // vesin never releases pre-allocated arrays
                vesin_free(&neighbors);
                check_cuda(cudaFree(pairs));
                check_cuda(cudaFree(shifts));
                check_cuda(cudaFree(distances));
                check_cuda(cudaFree(vectors));
            }
        }
    }

    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

static void CUDART_CB sleep_callback(void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

TEST_CASE("CUDA streams") {
    int32_t n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);
    check_cuda(cudaSetDevice(0));

    // 4x4x4 non-periodic lattice: with a cutoff of 1.1 there are 144 pairs
    // between nearest neighbors (288 in a full list). The "decoy" points are
    // too far apart to have any pair.
    auto points = std::vector<double>();
    auto decoy = std::vector<double>();
    for (size_t x = 0; x < 4; x++) {
        for (size_t y = 0; y < 4; y++) {
            for (size_t z = 0; z < 4; z++) {
                for (auto value : {x, y, z}) {
                    points.push_back(static_cast<double>(value));
                    decoy.push_back(10.0 * static_cast<double>(value));
                }
            }
        }
    }
    size_t n_points = points.size() / 3;
    double box[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    bool periodic[3] = {false, false, false};

    double (*d_points)[3] = nullptr;
    double (*d_box)[3] = nullptr;
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * points.size()));
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    // a non-blocking stream does not synchronize with the legacy default
    // stream, so vesin only sees the real points if it uses this stream
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (auto algorithm : {VesinBruteForce, VesinCellList}) {
        for (double skin : {0.0, 0.3}) {
            auto options = VesinOptions();
            options.cutoff = 1.1;
            options.full = true;
            options.algorithm = algorithm;
            options.skin = skin;
            options.return_shifts = true;
            options.cuda_stream = stream;

            auto neighbors = VesinNeighborList();
            const char* error_message = nullptr;
            auto compute = [&]() {
                auto status = vesin_neighbors(
                    d_points, n_points, d_box, d_periodic, {VesinCUDA, 0}, options, &neighbors, &error_message
                );
                REQUIRE(status == EXIT_SUCCESS);
            };

            // compile the kernels before the actual test, otherwise the
            // compilation can take longer than the delay below
            check_cuda(cudaMemcpy(d_points, decoy.data(), sizeof(double) * decoy.size(), cudaMemcpyHostToDevice));
            compute();
            CHECK(neighbors.length == 0);
            check_cuda(cudaStreamSynchronize(stream));

            // the real points only arrive after a delay on `stream`
            check_cuda(cudaLaunchHostFunc(stream, sleep_callback, nullptr));
            check_cuda(cudaMemcpyAsync(
                d_points, points.data(), sizeof(double) * points.size(), cudaMemcpyHostToDevice, stream
            ));

            compute();
            CHECK(neighbors.length == 288);

            check_cuda(cudaStreamSynchronize(stream));
            vesin_free(&neighbors);
        }
    }

    check_cuda(cudaStreamDestroy(stream));
    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

#else

TEST_CASE("CUDA tests are disabled") {}

#endif
